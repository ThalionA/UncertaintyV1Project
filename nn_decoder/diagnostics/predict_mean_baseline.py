# -*- coding: utf-8 -*-
"""Shuffle control, the simple way: how good is each trained decoder vs decoders
that have NO trial-specific information? (2026-06-10 meeting — "check how the loss
would be if we predict mean on each trial, or we kill weights from neurons"; the
shuffle-fit null added 2026-06-13 to compare all three nulls head-to-head.)

Three "no trial-information" nulls, on the matched loss_comparison_v1 cell:

  predict-mean  : output the marginal mean posterior (mean over trials of the IO
                  target) on EVERY trial. The OPTIMAL constant; the strictest null.
  kill-weights  : take the trained net, zero the input weights W_in and forward.
                  With W_in = 0 the output is input-independent — a single constant
                  = softmax(model(0)), identical in form for spatial (ppc) and
                  temporal (sampling). The net's own learned "no-neuron" fallback.
  shuffle-fit   : a net trained on label-shuffled data (saved per .mat as
                  Dist[arch+'_shf']) — the null cross_loss_eval uses for skill.

Empirically the ordering is predict-mean < kill-weights ≲ shuffle-fit: the shuffle
net can't beat the optimal constant (it overfits the scrambled labels and, under
peakiness-rewarding losses, emits confident-but-misplaced peaks rather than the
broad marginal), so it is a LOOSER null. We therefore normalise everything to
predict-mean (= 1.0, strictest) and show trained / kill-weights / shuffle ÷ that.

skill < 1 beats the strictest null; > 1 means worse than predicting the mean — the
headline being PCA's over-peaky posteriors falling below ALL three nulls under KL.

n = 6 mice; ratios are per-mouse then averaged. Loss maths == training, via
cross_loss_eval._eval_one.

Outputs (PNG+SVG) under figures/peakiness_scatter/:
  predict_mean_baseline.png   (2×2: arch × {own metric, KL}; 3 bars/loss ÷ predict-mean)

Usage:  python diagnostics/predict_mean_baseline.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from cross_loss_eval import _eval_one, _agg  # noqa: E402  (same maths as training)
from nn_classifier import SimpleFlexibleNNClassifier  # noqa: E402

LOSSES = ['PCA', 'CE', 'KL', 'JS', 'Wasserstein']
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
# the three nulls, all reported as loss / predict-mean loss (predict-mean = 1.0)
NULLS = [('tr', 'trained decoder', '#08519c'),
         ('kw', 'kill-weights', '#807dba'),
         ('sh', 'shuffle-fit', '#fd8d3c')]


def _slug(loss):
    return f'Q_{loss}_half_100ms' + ('_all' if loss == 'PCA' else '')


def _killweights_const(state_dict, model_params):
    """Bias-only prediction: softmax(model(0)) with W_in zeroed. Input-independent,
    so it is the net's constant fallback for BOTH archs. Returns (n_cats,) probs."""
    model = SimpleFlexibleNNClassifier(
        input_size=int(model_params['input_size']),
        hidden_sizes=list(np.atleast_1d(model_params['hidden_sizes'])),
        output_size=int(model_params['output_size']),
        activation=str(model_params.get('activation_function', 'relu')))
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        model.layers[0].weight.zero_()                       # kill the neuron weights
        logits = model(torch.zeros(1, int(model_params['input_size'])))
        return torch.softmax(logits, dim=-1).numpy().ravel()


def _load_killweights(cell, mk, split, arch):
    pt = cell / 'checkpoints' / f'{mk}_{split}.pt'
    if not pt.is_file():
        return None
    c = torch.load(str(pt), map_location='cpu', weights_only=False)
    if arch not in c or 'state_dict' not in c[arch]:
        return None
    return _killweights_const(c[arch]['state_dict'], c[arch]['model_params'])


def collect(results_root, run, split):
    """per (loss, arch): per-mouse ratios trained/predict-mean, kill-weights/PM,
    shuffle/PM under the OWN metric and KL; plus raw null losses for the table."""
    M = {}
    for loss in LOSSES:
        cell = Path(results_root) / run / _slug(loss)
        f = cell / f'{split}.mat'
        if not f.is_file():
            continue
        res = sio.loadmat(str(f), simplify_cells=True).get('results')
        if not isinstance(res, dict):
            continue
        for arch, _ in ARCHS:
            shf = arch + '_shf'
            rec = {f'{p}_{s}': [] for p in ('tr', 'kw', 'sh') for s in ('own', 'kl')}
            rec.update(pm_raw=[], kw_raw=[], sh_raw=[])
            for mk in sorted(res):
                D = res[mk]['Dist']
                pcs, evar = D.get('pcs'), D.get('explained_var')
                dec = np.asarray(D[arch]['decoded'], float)
                tgt = np.asarray(D[arch]['target'], float)
                ok = np.isfinite(tgt).all(1)
                if not ok.any():
                    continue
                # LEAVE-ONE-OUT marginal mean: predicting trial i from the OTHER
                # trials only. The plain test-set mean fits the null's free
                # parameter on the trials it is scored against, understating the
                # null loss by O(1/n) and so making "worse than chance" claims
                # easier — the anti-conservative direction here. 2026-07 audit.
                n_ok = int(ok.sum())
                tot = tgt[ok].sum(axis=0)
                pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
                if n_ok > 1:
                    pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
                kw_vec = _load_killweights(cell, mk, split, arch)
                kw = np.tile(kw_vec[None, :], (tgt.shape[0], 1)) if kw_vec is not None else None
                sdec = np.asarray(D[shf]['decoded'], float) if shf in D else None
                stgt = np.asarray(D[shf]['target'], float) if shf in D else tgt
                for metric, suf in ((loss, 'own'), ('KL', 'kl')):
                    pml = _eval_one(pm, tgt, metric, pcs, evar)
                    if not (np.isfinite(pml) and pml > 0):
                        continue
                    trl = _eval_one(dec, tgt, metric, pcs, evar)
                    kwl = _eval_one(kw, tgt, metric, pcs, evar) if kw is not None else np.nan
                    shl = _eval_one(sdec, stgt, metric, pcs, evar) if sdec is not None else np.nan
                    if np.isfinite(trl):
                        rec[f'tr_{suf}'].append(trl / pml)
                    if np.isfinite(kwl):
                        rec[f'kw_{suf}'].append(kwl / pml)
                    if np.isfinite(shl):
                        rec[f'sh_{suf}'].append(shl / pml)
                    if suf == 'own':
                        rec['pm_raw'].append(pml)
                        if np.isfinite(kwl):
                            rec['kw_raw'].append(kwl)
                        if np.isfinite(shl):
                            rec['sh_raw'].append(shl)
            M[(loss, arch)] = rec
    return M


def main(results_root, run, split, out_root):
    ps.apply()
    M = collect(results_root, run, split)
    if not M:
        raise SystemExit('no loss cells found.')

    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2), sharex=True)
    x = np.arange(len(LOSSES))
    w = 0.26
    cols = [('own', "judged on each loss's OWN metric"),
            ('kl', 'judged on the KL metric (calibrated)')]
    for r, (arch, alabel) in enumerate(ARCHS):
        for c, (suf, ctitle) in enumerate(cols):
            ax = axes[r, c]
            for k, (pre, lab, col) in enumerate(NULLS):
                means, sems = [], []
                for loss in LOSSES:
                    m, s = _agg(M.get((loss, arch), {}).get(f'{pre}_{suf}', []))
                    means.append(m); sems.append(s)
                ax.bar(x + (k - 1) * w, means, w, yerr=sems, capsize=2,
                       color=col, label=lab)
            ax.axhline(1.0, color='k', ls='--', lw=1.1)
            ax.set_xticks(x); ax.set_xticklabels(ps.loss_labels(LOSSES), rotation=20, ha='right')
            if c == 0:
                ax.set_ylabel(f'{alabel}\nloss / predict-mean loss', fontsize=8.5)
            if r == 0:
                ax.set_title(ctitle, fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='upper left', ncol=1)
                ax.text(len(LOSSES) - 0.5, 1.0, 'predict-mean (strictest null)',
                        ha='right', va='bottom', fontsize=6.5, color='0.3')

    ps.label_panels(axes.ravel())
    fig.suptitle('Trained decoder vs three "no trial-info" nulls — all ÷ predict-mean '
                 '(1.0 = predict-mean; <1 beats it, ≥1 worse; Q/half/100ms, 6 mice)',
                 y=1.02)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'predict_mean_baseline')

    # numeric: the null ordering (predict-mean < kill-weights ≲ shuffle) + skills.
    print('null raw losses (own metric) and ratios to predict-mean (mean / 6 mice)\n')
    hdr = (f"{'loss':12s}{'arch':9s} {'PM_raw':>9s} {'KW/PM':>7s} {'SH/PM':>7s} | "
           f"{'trained/PM':>11s} {'(KL) tr/PM':>11s} {'KW/PM':>7s} {'SH/PM':>7s}")
    print(hdr); print('-' * len(hdr))
    for loss in LOSSES:
        for arch, alabel in ARCHS:
            r = M.get((loss, arch), {})
            pm = _agg(r.get('pm_raw', []))[0]
            kwpm = _agg(r.get('kw_raw', []))[0] / pm if pm else np.nan
            shpm = _agg(r.get('sh_raw', []))[0] / pm if pm else np.nan
            tro = _agg(r.get('tr_own', []))[0]
            trk = _agg(r.get('tr_kl', []))[0]
            kwk = _agg(r.get('kw_kl', []))[0]
            shk = _agg(r.get('sh_kl', []))[0]
            print(f"{loss:12s}{alabel:9s} {pm:9.3g} {kwpm:6.2f}x {shpm:6.2f}x | "
                  f"{tro:10.2f}x {trk:10.2f}x {kwk:6.2f}x {shk:6.2f}x")
    print(f'\nDone. {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', default='loss_comparison_v1')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/peakiness_scatter')
    a = ap.parse_args()
    main(a.results_root, a.run, a.split, a.out_root)
