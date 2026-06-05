# -*- coding: utf-8 -*-
"""Weight evolution for the three PCA-loss variants + the weight→peakiness law
(2026-06-03, Theo's question: do weights need to be extreme for peaky outputs?
is it linear/nonlinear?).

For each variant (evar / flat-evar / PCA+shape λ=10) we read the per-epoch weight
snapshots already saved in the `wm3*` checkpoints (the runs forced
`track_training_history` + `weight_snapshot_every=10`), rebuild the decoder at
each snapshot, forward the test set to get the OUTPUT LOGITS, and relate:
  - ‖W_out‖ (raw and /√H) vs epoch — does the peaky variant carry bigger weights?
  - decoded max-prob vs the logit spread (std of logits per trial) — the softmax
    law: peakiness is a NONLINEAR (saturating) function of the logit scale, and
    the logit scale is linear in the weights.

Reuses `plot_weight_evolution_cell` keys + `decode_entropy_trajectory._build_model`.

Outputs under figures/loss_variants/weights/:
  wout_vs_epoch_<arch>.png      ‖W_out‖ raw + /√H vs epoch, 3 variants
  peakiness_vs_weight_<arch>.png  max-prob vs ‖W_out‖ and vs logit-spread

Usage:  python weight_evolution_variants.py            # spat (PPC, no entropy pen.)
        python weight_evolution_variants.py --arch temp
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

import peakiness_style as ps
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostics.decode_entropy_trajectory import _build_model, _as_np  # noqa: E402

VARIANTS = [('PCA evar', 'wm3'), ('flat-evar', 'wm3_flatevar'),
            ('PCA+shape λ=10', 'wm3_shape10')]
SLUG = 'Q_PCA_half_100ms_all'
KEY_WOUT = 'layers.1.weight'


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def collect(results_root, run, split, arch, activation):
    """Per-snapshot, across-mouse means: epoch, ‖W_out‖, ‖W_out‖/√H, logit_std,
    max_prob. Reads weight snapshots + X_test straight from the checkpoints."""
    ck_dir = Path(results_root) / run / SLUG / 'checkpoints'
    pts = sorted(ck_dir.glob(f'mouse_*_{split}.pt'))
    per_mouse = []
    for pt in pts:
        ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        if arch not in ck:
            continue
        sub = ck[arch]
        hist = sub.get('history')
        if not hist or hist.get('state_dicts') is None:
            continue
        eps = np.asarray(hist['snapshot_epochs'], int)
        X = np.asarray(sub['X_test'], np.float32)            # (n, T, neurons)
        rows = []
        for sd in hist['state_dicts']:
            Wout = _as_np(sd[KEY_WOUT])                       # (cats, H)
            H = Wout.shape[1]
            model = _build_model(sd, activation)
            xb = torch.tensor(X)
            with torch.no_grad():
                logits = (model(xb.mean(1)) if arch == 'spat' else model(xb)).numpy()
            if arch == 'temp':                               # per-bin → trial via Jensen
                probs = _softmax(logits).mean(1)             # (n, cats)
                logit_std = logits.std(-1).mean()            # per-bin logit spread
            else:
                probs = _softmax(logits)
                logit_std = logits.std(-1).mean()
            rows.append((np.linalg.norm(Wout), np.linalg.norm(Wout) / np.sqrt(H),
                         logit_std, probs.max(-1).mean()))
        per_mouse.append((eps, np.asarray(rows)))            # rows: (S,4)
    if not per_mouse:
        return None
    grid = min((e for e, _ in per_mouse), key=len)
    M = np.full((len(per_mouse), len(grid), 4), np.nan)
    for i, (e, r) in enumerate(per_mouse):
        idx = {int(x): j for j, x in enumerate(e)}
        for j, x in enumerate(grid):
            if int(x) in idx:
                M[i, j] = r[idx[int(x)]]
    return grid, np.nanmean(M, axis=0)   # (S,), (S,4): wout,wout_fanin,logit_std,maxprob


def main(results_root, split, arch, out_root):
    ps.apply()
    data = {}
    for name, run in VARIANTS:
        cfg = Path(results_root) / run / SLUG / f'{split}.mat'
        act = 'tanh'
        if cfg.is_file():
            act = sio.loadmat(str(cfg), simplify_cells=True)['config'].get('activation_function', 'tanh')
        r = collect(results_root, run, split, arch, act)
        if r is not None:
            data[name] = r
            ep, A = r
            print(f'  {name:16s} {arch}: ‖W_out‖ {A[0,0]:.2f}→{A[-1,0]:.2f}  '
                  f'logit_std {A[0,2]:.2f}→{A[-1,2]:.2f}  max-prob {A[0,3]:.3f}→{A[-1,3]:.3f}')
    if not data:
        raise SystemExit('no snapshots found.')
    out_dir = Path(out_root) / 'weights'

    # Fig A: ‖W_out‖ raw and /√H vs epoch
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (ep, A) in data.items():
        axes[0].plot(ep, A[:, 0], color=ps.color(name), lw=2.2, marker='o', ms=3, label=name)
        axes[1].plot(ep, A[:, 1], color=ps.color(name), lw=2.2, marker='o', ms=3, label=name)
    axes[0].set_ylabel('‖W_out‖ (raw)'); axes[1].set_ylabel('‖W_out‖ / √H')
    for ax in axes:
        ax.set_xlabel('epoch'); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'Output-weight norm vs training — {arch.upper()} (3 PCA variants)', y=1.02)
    fig.tight_layout(); _save(fig, out_dir, f'wout_vs_epoch_{arch}')

    # Fig B: peakiness vs ‖W_out‖ and vs logit spread (the softmax law)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (ep, A) in data.items():
        axes[0].plot(A[:, 0], A[:, 3], color=ps.color(name), lw=1.5, marker='o', ms=4, label=name)
        axes[1].plot(A[:, 2], A[:, 3], color=ps.color(name), lw=1.5, marker='o', ms=4, label=name)
    # target peakiness reference
    tgt_mp = _target_maxprob(results_root, split, arch)
    for ax in axes:
        ax.axhline(tgt_mp, ls='--', color='k', lw=1.2, label=f'IO target ({tgt_mp:.3f})')
        ax.set_ylabel('decoded max-probability'); ax.legend(frameon=False, fontsize=8)
    axes[0].set_xlabel('‖W_out‖ (raw)')
    axes[1].set_xlabel('logit spread  (std of logits per trial)')
    fig.suptitle(f'Peakiness vs weights — {arch.upper()}  '
                 '(variants do not collapse → not set by ‖W_out‖ alone)', y=1.02)
    fig.tight_layout(); _save(fig, out_dir, f'peakiness_vs_weight_{arch}')

    fig_logit_profiles(results_root, split, arch, out_dir)


def fig_logit_profiles(results_root, split, arch, out_dir, mouse='mouse_0', n=4):
    """Final-model output LOGIT profiles z (centred) for example trials, per
    variant. Shows WHY equal-norm weights give different peakiness: evar makes a
    concentrated spike, shape makes a smooth high-amplitude bump (same scale,
    broad softmax)."""
    # load final state_dict + X_test + activation per variant (mouse_0)
    prof = {}
    Xref = None
    for name, run in VARIANTS:
        pt = Path(results_root) / run / SLUG / 'checkpoints' / f'{mouse}_{split}.pt'
        if not pt.is_file():
            continue
        ck = torch.load(str(pt), map_location='cpu', weights_only=False)
        if arch not in ck:
            continue
        sub = ck[arch]
        sd = sub['history']['state_dicts'][-1]
        cfg = Path(results_root) / run / SLUG / f'{split}.mat'
        act = sio.loadmat(str(cfg), simplify_cells=True)['config'].get('activation_function', 'tanh') \
            if cfg.is_file() else 'tanh'
        model = _build_model(sd, act)
        X = np.asarray(sub['X_test'], np.float32); Xref = X
        with torch.no_grad():
            z = model(torch.tensor(X).mean(1)).numpy() if arch == 'spat' else \
                model(torch.tensor(X)).numpy().mean(1)
        prof[name] = z
    if not prof or Xref is None:
        return
    # pick trials with diverse decoded peaks under evar
    z0 = prof.get('PCA evar')
    order = np.argsort(np.argmax(z0, 1)) if z0 is not None else np.arange(Xref.shape[0])
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    x = np.arange(z0.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.0), sharex=True, sharey=True)
    for ax, tr in zip(axes, picks):
        for name, z in prof.items():
            ax.plot(x, z[tr] - z[tr].mean(), color=ps.color(name), lw=1.6, label=name)
        ax.set_xlabel('orientation bin'); ax.set_title(f'trial {tr}', fontsize=9)
        ax.axhline(0, color='0.7', lw=0.6)
    axes[0].set_ylabel('output logit  z − mean'); axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f'Output logit profiles, final model — {arch.upper()}  '
                 '(evar spikes; shape is a smooth bump at the same scale)', y=1.04)
    fig.tight_layout(); _save(fig, out_dir, f'logit_profiles_{arch}')


def _target_maxprob(results_root, split, arch):
    m = sio.loadmat(f'{results_root}/wm3/{SLUG}/{split}.mat', simplify_cells=True)['results']
    return float(np.mean([np.asarray(m[mk]['Dist'][arch]['target'], float).max(1).mean() for mk in m]))


def _save(fig, out_dir, stem):
    ps.save_fig(fig, out_dir, stem)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--arch', default='spat', choices=('spat', 'temp'))
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_variants')
    a = ap.parse_args()
    main(a.results_root, a.split, a.arch, a.out_root)
