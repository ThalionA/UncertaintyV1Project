# -*- coding: utf-8 -*-
"""Decoded-posterior entropy vs training epoch — why does the PCA loss go peaky?

Tests the mechanistic claim (2026-06-03, answering Theo's "what *favours* peaky?"):
the PCA-weighted L2 loss has its global minimum at pred=target (broad), so
peakiness is NOT a loss-optimum property. The story is a trajectory one — from a
near-uniform init the net concentrates mass to match the heavily-weighted
position PCs (overshooting into peaky, since that's the gradient-efficient way to
hit position), the width direction has ~0 gradient so it never broadens back, and
early stopping freezes it mid-flight. KL/JS can't sit peaky (unbounded penalty for
pred->0 where target>0) so they stay calibrated.

Prediction if that's right: decoded entropy should DROP BELOW the target entropy
early under PCA and stay there through the best-val (early-stop) epoch, while
KL/JS sit near the target entropy.

We have everything on disk: the `loss_comparison_v1` checkpoints store weight
snapshots (`history['state_dicts']` at `snapshot_epochs`) AND the test set
(`X_test`). So we re-decode X_test at each snapshot, mirroring the eval forward
(PPC: softmax(MLP(mean_t x)); SBC: mean_t softmax(MLP(x))), and track the
across-trial mean entropy of the decoded posterior vs epoch.

Output: figures/loss_sweep_plots/<run>/entropy_trajectory/
  entropy_vs_epoch_<arch>.png   one line per loss; dashed = target entropy;
                                star = best-val epoch (as deployed)

Usage
-----
    python diagnostics/decode_entropy_trajectory.py
    python diagnostics/decode_entropy_trajectory.py --target L --mouse all
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
import decoder_plotting_utils as dpu  # noqa: E402
from nn_classifier import SimpleFlexibleNNClassifier  # noqa: E402

LOSSES = ('PCA', 'CE', 'KL', 'JS', 'Wasserstein')
LOSS_COLOR = {'PCA': '#e6550d', 'CE': '#008837', 'KL': '#7b3294',
              'JS': '#3690c0', 'Wasserstein': '#a6611a'}
KEY_WIN, KEY_WOUT = 'layers.0.weight', 'layers.1.weight'


def _slug(target, loss, window, bin_ms):
    return f'{target}_{loss}_{window}_{bin_ms}ms' + ('_all' if loss == 'PCA' else '')


def _as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def _entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _reduce(p, metric):
    """Across-trial mean of the per-trial peakiness measure. For the spatial
    (PPC) decoder the outputs are jagged/multimodal, so entropy understates the
    peakiness — max-probability is the sharper read there."""
    per = np.max(p, axis=-1) if metric == 'maxprob' else _entropy(p)
    return float(np.mean(per))


def _ylabel(metric):
    return ('decoded mean max-probability' if metric == 'maxprob'
            else 'decoded posterior entropy  H  [nats]')


def _build_model(sd, activation):
    n_in = _as_np(sd[KEY_WIN]).shape[1]
    hidden = _as_np(sd[KEY_WIN]).shape[0]
    n_out = _as_np(sd[KEY_WOUT]).shape[0]
    m = SimpleFlexibleNNClassifier(n_in, [hidden], n_out, activation=activation)
    m.load_state_dict({k: (v if torch.is_tensor(v) else torch.tensor(np.asarray(v)))
                       for k, v in sd.items()})
    m.eval()
    return m


def _decode(model, X, arch):
    """Mirror evaluate forward. X: (B, T, n_neurons). Returns (B, n_cats)."""
    xb = torch.tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        if arch == 'spat':                              # PPC: average then decode
            probs = torch.softmax(model(xb.mean(dim=1)), dim=-1)
        else:                                           # SBC: decode per bin then average
            probs = torch.softmax(model(xb), dim=-1).mean(dim=1)
    return probs.numpy()


def load_cell(results_root, run_name, target, loss, window, bin_ms, split, mice, activation,
              metric='entropy'):
    """Per-mouse {epochs, decoded_entropy(epochs), target_entropy, best} for one
    (loss, arch) — keyed by arch. Re-decodes X_test at each weight snapshot."""
    ck_dir = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / 'checkpoints'
    if not ck_dir.is_dir():
        return {}
    pts = sorted(ck_dir.glob(f'mouse_*_{split}.pt'))
    if mice is not None:
        pts = [p for p in pts if any(f'mouse_{m}_' in p.name for m in mice)]
    out = {'spat': [], 'temp': []}
    for pt in pts:
        ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        for arch in ('spat', 'temp'):
            if arch not in ck or not isinstance(ck[arch], dict):
                continue
            sub = ck[arch]
            hist = sub.get('history')
            if not hist or hist.get('state_dicts') is None:
                continue
            eps = np.asarray(hist['snapshot_epochs'], dtype=int)
            X = sub['X_test']
            tgt = _as_np(sub.get('history', {})) if False else None  # placeholder
            ent = []
            for sd in hist['state_dicts']:
                model = _build_model(sd, activation)
                dec = _decode(model, X, arch)
                ent.append(_reduce(dec, metric))
            # best-val epoch (1-indexed onto the epoch axis of the curves)
            best = None
            for k in ('val_total_loss', 'val_fit_loss'):
                v = hist.get(k)
                if v is not None and len(np.asarray(v)):
                    best = int(np.nanargmin(np.asarray(v, float)))
                    break
            out[arch].append({'epochs': eps, 'entropy': np.asarray(ent), 'best': best})
    return out


def target_entropy(results_root, run_name, target, loss, window, bin_ms, split, mice,
                   metric='entropy'):
    """Mean across-trial entropy of the IO target posteriors (same per arch)."""
    mat = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / f'{split}.mat'
    if not mat.is_file():
        return None
    d = sio.loadmat(str(mat), simplify_cells=True)['results']
    keys = [f'mouse_{m}' for m in mice] if mice is not None else sorted(d.keys())
    vals = []
    for mk in keys:
        if mk in d:
            tg = np.asarray(d[mk]['Dist']['spat']['target'], float)
            vals.append(_reduce(tg, metric))
    return float(np.mean(vals)) if vals else None


def _interp_mean(curves):
    """Mean entropy curve over mice on the shortest shared snapshot grid."""
    grids = [c['epochs'] for c in curves]
    common = grids[int(np.argmin([len(g) for g in grids]))]
    M = np.full((len(curves), len(common)), np.nan)
    for i, c in enumerate(curves):
        idx = {int(e): j for j, e in enumerate(c['epochs'])}
        for j, e in enumerate(common):
            if int(e) in idx:
                M[i, j] = c['entropy'][idx[int(e)]]
    return common, np.nanmean(M, axis=0)


def overlay_runs(main_run, compare_runs, labels, target, window, bin_ms, split,
                 mice, activation, results_root, out_root, tgt_ref, metric='entropy'):
    """Overlay the PCA peakiness trajectory across runs (e.g. evar-weighted vs
    flat-evar) on one axis per arch, with the target line + a calibrated
    reference (mean of CE/KL/JS from the main run). The headline control plot."""
    runs = [main_run] + list(compare_runs)
    # labels (if given) map one-per-run in order [main, compare...]; else auto.
    lbls = list(labels) if labels and len(labels) == len(runs) else \
        [f'PCA · {r}' for r in runs]
    styles = ['-', '--', ':', '-.']
    suffix = '_maxprob' if metric == 'maxprob' else ''
    out_dir = Path(out_root) / main_run / 'entropy_trajectory'
    for arch in ('spat', 'temp'):
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        # calibrated reference from the main run (CE/KL/JS mean final)
        cal = []
        for cl in ('CE', 'KL', 'JS'):
            c = load_cell(results_root, main_run, target, cl, window, bin_ms, split, mice, activation, metric)
            if c.get(arch):
                xs, m = _interp_mean(c[arch]); cal.append(m[-1])
        for i, (r, lab) in enumerate(zip(runs, lbls)):
            c = load_cell(results_root, r, target, 'PCA', window, bin_ms, split, mice, activation, metric)
            if not c.get(arch):
                continue
            xs, m = _interp_mean(c[arch])
            ax.plot(xs, m, color='#e6550d', ls=styles[i % len(styles)], lw=2.6,
                    marker='o', ms=3, label=f'{lab}  (→{m[-1]:.3f})')
        if tgt_ref is not None:
            ax.axhline(tgt_ref, ls='--', lw=1.6, color='k', label=f'IO target ({tgt_ref:.3f})')
        if cal:
            ax.axhline(np.mean(cal), ls='-.', lw=1.3, color='#3690c0',
                       label=f'CE/KL/JS calibrated ({np.mean(cal):.3f})')
        ax.set_xlabel('epoch (weight snapshot)')
        ax.set_ylabel(_ylabel(metric))
        pen = 'no entropy penalty' if arch == 'spat' else 'has entropy penalty'
        ax.set_title(f'PCA {metric}: evar-weighted vs flat-evar — {arch.upper()} ({pen})\n'
                     f'{target} {window} {bin_ms}ms {split}; flattening evar removes the over-sharpening')
        ax.legend(frameon=False, fontsize=8.5)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(out_dir / f'entropy_compare_{arch}{suffix}.{ext}', bbox_inches='tight', dpi=140)
        plt.close(fig)
        print(f'  -> entropy_compare_{arch}{suffix}.png/.svg')


def main(run_name, target, window, bin_ms, split, mouse_sel, results_root, out_root,
         compare_runs=None, compare_labels=None, metric='entropy'):
    dpu.set_style()
    mice = None if mouse_sel == 'all' else [int(mouse_sel)]
    suffix = '_maxprob' if metric == 'maxprob' else ''
    # activation from the cell config (all cells share it).
    cfg_mat = Path(results_root) / run_name / _slug(target, 'KL', window, bin_ms) / f'{split}.mat'
    activation = 'tanh'
    if cfg_mat.is_file():
        activation = sio.loadmat(str(cfg_mat), simplify_cells=True)['config'].get(
            'activation_function', 'tanh')
    info = f'{target} {window} {bin_ms}ms {split} ' + \
           ('all mice' if mice is None else f'mouse {mice[0]}')
    print(f'{metric} trajectory: {run_name} | {info} | activation={activation}')

    by_loss = {l: load_cell(results_root, run_name, target, l, window, bin_ms, split, mice, activation, metric)
               for l in LOSSES}
    tgt_H = {l: target_entropy(results_root, run_name, target, l, window, bin_ms, split, mice, metric)
             for l in LOSSES}
    tgt_ref = np.nanmean([v for v in tgt_H.values() if v is not None]) if any(tgt_H.values()) else None
    n_cats = 91
    ceiling = (1.0 / n_cats) if metric == 'maxprob' else np.log(n_cats)  # uniform reference

    out_dir = Path(out_root) / run_name / 'entropy_trajectory'
    for arch in ('spat', 'temp'):
        curves = {l: by_loss[l].get(arch, []) for l in LOSSES}
        if not any(curves.values()):
            print(f'  [skip] {arch}: no snapshots'); continue
        fig, ax = plt.subplots(figsize=(8.5, 5.4))
        for l in LOSSES:
            cs = curves[l]
            if not cs:
                continue
            xs, m = _interp_mean(cs)
            ax.plot(xs, m, color=LOSS_COLOR[l], lw=2.4, marker='o', ms=3.5, label=l)
            bests = [c['best'] for c in cs if c['best'] is not None]
            if bests:
                be = int(round(np.mean(bests)))
                yi = np.interp(be, xs, m)
                ax.scatter([be], [yi], color=LOSS_COLOR[l], marker='*', s=160,
                           zorder=6, edgecolor='k', lw=0.6)
        if tgt_ref is not None:
            ax.axhline(tgt_ref, ls='--', lw=1.6, color='k',
                       label=f'IO target ({tgt_ref:.3f})')
        ax.axhline(ceiling, ls=':', lw=1, color='0.5',
                   label=f'uniform ({ceiling:.3f})')
        ax.set_xlabel('epoch (weight snapshot)')
        ax.set_ylabel(_ylabel(metric) + '  (mean over trials)')
        side = 'above' if metric == 'maxprob' else 'below'
        pen = 'no entropy penalty' if arch == 'spat' else 'has entropy penalty'
        ax.set_title(f'Decoded {metric} vs training — {arch.upper()} ({pen})  ({info})\n'
                     f'{side} the dashed line = peakier than the target;  '
                     'star = best-val (deployed) epoch')
        ax.legend(frameon=False, fontsize=8.5, ncol=2)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(out_dir / f'entropy_vs_epoch_{arch}{suffix}.{ext}', bbox_inches='tight', dpi=140)
        plt.close(fig)
        print(f'  -> entropy_vs_epoch_{arch}{suffix}.png/.svg')
        # quick numeric readout
        for l in LOSSES:
            cs = curves[l]
            if not cs:
                continue
            xs, m = _interp_mean(cs)
            print(f'     {l:12s} {arch}: init={m[0]:.3f} -> final={m[-1]:.3f}'
                  f'  (target {tgt_ref:.3f})' if tgt_ref else '')
    if compare_runs:
        overlay_runs(run_name, compare_runs, compare_labels, target, window,
                     bin_ms, split, mice, activation, results_root, out_root, tgt_ref, metric)
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--mouse', default='0')
    ap.add_argument('--compare-runs', nargs='*', default=None,
                    help='extra run-names whose PCA curve to overlay vs the main run')
    ap.add_argument('--compare-labels', nargs='*', default=None,
                    help='legend labels for the overlay (main run first is auto)')
    ap.add_argument('--metric', default='entropy', choices=('entropy', 'maxprob'),
                    help='peakiness measure; maxprob is the sharper read for the '
                         'jagged spatial (PPC) decoder')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(a.run_name, a.target, a.window, a.bin_ms, a.split, a.mouse,
         a.results_root, a.out_root, compare_runs=a.compare_runs,
         compare_labels=a.compare_labels, metric=a.metric)
