# -*- coding: utf-8 -*-
"""Toy: the width-matched fix + WHY the PCA decoder keeps getting peakier (2026-06-03).

Two things in the same controlled toy world (`toy_peakiness_model`):

1. THE FIX (request 1). A width-matched loss = the evar-weighted PCA loss PLUS a
   small unweighted-L2 (Brier) term, i.e. floor the per-PC weights so the shape
   subspace is no longer free. Keeps PCA's location emphasis; adds a restoring
   force on width. Shown in the phase portrait + example posteriors: it climbs to
   the target peakiness and STOPS, unlike plain PCA.

2. THE WHY (request 2). Does over-sharpening actually reduce the loss, and would
   earlier stopping or a lower learning rate help? We log, per epoch, the TRAIN and
   TEST weighted-PCA loss alongside peakiness. The answer is an overfitting answer:
   after location is matched the TRAIN loss keeps creeping down (the net fits noise
   into the loss-blind shape subspace → overconfident, peaky) while the TEST loss
   flattens/rises — and peakiness tracks the train–test gap. Early stopping at the
   best-TEST epoch does cut the peakiness (it is overfitting), but the weighted
   loss is a weak stop signal because it barely sees the shape subspace; a lower LR
   only slows the same drift.

Outputs under figures/toy_peakiness/:
  widthmatched_phase.png     phase portrait: PCA vs KL vs PCA+shape (+ flat-L2)
  widthmatched_examples.png  example posteriors: target vs PCA vs PCA+shape
  why_peakier.png            PCA train/test loss + peakiness vs epoch; best-test
                             (early-stop) marker; a lower-LR peakiness overlay

Usage:  python diagnostics/toy_width_matched.py
        python diagnostics/toy_width_matched.py --lam 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
from diagnostics.toy_peakiness_model import make_data, MLP  # noqa: E402

def _entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -np.sum(p * np.log(p), -1)


def compute_loss(pred, target, kind, pcs, evar, lam):
    if kind == 'KL':
        return torch.sum(target * (torch.log(target + 1e-12) - torch.log(pred + 1e-12)), -1).mean()
    brier = ((pred - target) ** 2).sum(-1).mean()
    if kind == 'flatL2':
        return brier
    pca = (evar * ((pred @ pcs.T - target @ pcs.T) ** 2)).sum(-1).mean() * 100.0
    if kind == 'PCA':
        return pca
    if kind == 'PCAshape':                              # width-matched: floor the weights
        return pca + lam * brier
    raise ValueError(kind)


def weighted_pca_loss_np(pred, target, pcs, evar):
    """The evar-weighted PCA loss, numpy, for tracking train/test."""
    e = ((pred @ pcs.T - target @ pcs.T) ** 2 * evar).sum(-1).mean() * 100.0
    return float(e)


def train_logged(Xtr, Ttr, Xte, Tte, kind, pcs, evar, lam, hidden, epochs,
                 log_every, lr, seed):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], hidden, Ttr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(Xtr); T = torch.tensor(Ttr)
    pcs_t = torch.tensor(pcs); evar_t = torch.tensor(evar)
    kloc = int(np.searchsorted(np.cumsum(evar), 0.90)) + 1
    log = {'epoch': [], 'loc': [], 'shape': [], 'maxprob': [],
           'train_loss': [], 'test_loss': []}

    def snap(ep):
        with torch.no_grad():
            dtr = torch.softmax(model(X), -1).numpy()
            dte = torch.softmax(model(torch.tensor(Xte)), -1).numpy()
        err = ((dte @ pcs.T - Tte @ pcs.T) ** 2).mean(0)
        log['epoch'].append(ep)
        log['loc'].append(err[:kloc].sum())
        log['shape'].append(err[kloc:].sum())
        log['maxprob'].append(dte.max(1).mean())
        log['train_loss'].append(weighted_pca_loss_np(dtr, Ttr, pcs, evar))
        log['test_loss'].append(weighted_pca_loss_np(dte, Tte, pcs, evar))

    snap(0)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        compute_loss(torch.softmax(model(X), -1), T, kind, pcs_t, evar_t, lam).backward()
        opt.step()
        if ep % log_every == 0:
            snap(ep)
    return {k: np.asarray(v) for k, v in log.items()}


def main(C, n_train, n_test, n_neurons, tune_width, noise, target_width,
         hidden, epochs, log_every, lam, seed, out_root):
    ps.apply()
    Xtr, Ttr, _ = make_data(n_train, C, n_neurons, tune_width, noise, target_width, seed)
    Xte, Tte, _ = make_data(n_test, C, n_neurons, tune_width, noise, target_width, seed + 1)
    pca = PCA().fit(Ttr)
    pcs = pca.components_.astype(np.float32); evar = pca.explained_variance_ratio_.astype(np.float32)
    tgt_mp = Tte.max(1).mean()
    print(f'lam={lam}; target max-prob={tgt_mp:.3f}')

    variants = {'PCA (evar-weighted)': 'PCA', 'KL': 'KL',
                'flat L2 (Brier)': 'flatL2', 'PCA + shape term': 'PCAshape'}
    logs = {name: train_logged(Xtr, Ttr, Xte, Tte, kind, pcs, evar, lam, hidden,
                               epochs, log_every, 1e-2, seed)
            for name, kind in variants.items()}
    for name, L in logs.items():
        print(f'  {name:20s}: max-prob {L["maxprob"][0]:.3f}->{L["maxprob"][-1]:.3f}  '
              f'(target {tgt_mp:.3f})')

    out_dir = Path(out_root)
    _fig_phase(logs, tgt_mp, out_dir)
    _fig_examples(Xte, Tte, Xtr, Ttr, pcs, evar, lam, hidden, epochs, seed, out_dir)
    # WHY figure: PCA train/test loss + peakiness, + a lower-LR peakiness overlay
    pca_lowlr = train_logged(Xtr, Ttr, Xte, Tte, 'PCA', pcs, evar, lam, hidden,
                             epochs, log_every, 1e-3, seed)
    _fig_why(logs['PCA (evar-weighted)'], pca_lowlr, tgt_mp, out_dir)
    print(f'Done. {out_dir.resolve()}')


def _fig_phase(logs, tgt_mp, out_dir):
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    for name, L in logs.items():
        ax.plot(L['loc'], L['maxprob'], color=ps.color(name), lw=1.6, alpha=0.85, label=name)
        ax.scatter(L['loc'][-1], L['maxprob'][-1], color=ps.color(name), s=80, edgecolor='k', lw=0.7, zorder=4)
    ax.axhline(tgt_mp, ls='--', color='k', lw=1.2, label=f'target ({tgt_mp:.3f})')
    ax.set_xlabel('location-subspace error'); ax.set_ylabel('peakiness (max-prob)')
    ax.set_title('Width-matched loss: peakiness vs location error')
    ax.legend(frameon=False, fontsize=8.5, loc='best')
    _save(fig, out_dir, 'widthmatched_phase')


def _fig_examples(Xte, Tte, Xtr, Ttr, pcs, evar, lam, hidden, epochs, seed, out_dir):
    # retrain PCA and PCA+shape to get final decoded posteriors
    def final(kind):
        L = train_logged(Xtr, Ttr, Xte, Tte, kind, pcs, evar, lam, hidden, epochs, epochs, 1e-2, seed)  # noqa
        torch.manual_seed(seed); m = MLP(Xtr.shape[1], hidden, Ttr.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=1e-2); X = torch.tensor(Xtr); T = torch.tensor(Ttr)
        pcs_t = torch.tensor(pcs); evar_t = torch.tensor(evar)
        for _ in range(epochs):
            opt.zero_grad(); compute_loss(torch.softmax(m(X), -1), T, kind, pcs_t, evar_t, lam).backward(); opt.step()
        with torch.no_grad():
            return torch.softmax(m(torch.tensor(Xte)), -1).numpy()
    dp = final('PCA'); ds = final('PCAshape')
    rng = np.random.default_rng(1); picks = rng.choice(Tte.shape[0], 4, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=ps.figsize(4, 1), sharey=True)
    x = np.arange(Tte.shape[1])
    for ax, tr in zip(axes, picks):
        ps.target_band(ax, x, Tte[tr], label='target')
        ax.plot(x, dp[tr], color=ps.PCA_EVAR, lw=1.8, label='PCA')
        ax.plot(x, ds[tr], color=ps.SHAPE, lw=1.8, label='PCA + shape')
        ax.set_xlabel('orientation bin'); ax.set_title(f'trial {tr}', fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc='best'); axes[0].set_ylabel('probability')
    ps.label_panels(axes)
    fig.suptitle('Width-matched loss: example posteriors', y=1.02)
    _save(fig, out_dir, 'widthmatched_examples')


def _fig_why(L, L_lowlr, tgt_mp, out_dir):
    """(a) PCA train vs test weighted loss (left axis) and peakiness (right axis)
    vs epoch; best-test epoch starred = where early stopping would fire; lower-LR
    peakiness dashed = same drift, slower. (b) peakiness vs the train–test gap, one
    point per epoch coloured by epoch — they rise together (Pearson r), the direct
    overfitting signature."""
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    # (a) trajectories
    ax = axes[0]
    ax.plot(L['epoch'], L['train_loss'], color='#666666', lw=2.2, label='train loss')
    ax.plot(L['epoch'], L['test_loss'], color='#000000', lw=2.2, ls='--', label='test loss')
    best = int(L['epoch'][np.argmin(L['test_loss'])])
    ax.axvline(best, color='#2166ac', lw=1.3, ls=':', label=f'best-test ({best}) = early stop')
    ax.set_xlabel('epoch'); ax.set_ylabel('weighted PCA loss')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax2 = ax.twinx()
    ax2.plot(L['epoch'], L['maxprob'], color=ps.PCA_EVAR, lw=2.4, label='peakiness (lr=1e-2)')
    ax2.plot(L_lowlr['epoch'], L_lowlr['maxprob'], color=ps.PCA_EVAR,
             lw=1.8, ls='--', alpha=0.7, label='peakiness (lr=1e-3)')
    ax2.axhline(tgt_mp, color='0.5', ls=':', lw=1)
    mp_best = L['maxprob'][np.argmin(L['test_loss'])]
    ax2.scatter([best], [mp_best], color=ps.PCA_EVAR, marker='*', s=180,
                edgecolor='k', lw=0.6, zorder=5)
    ax2.set_ylabel('peakiness (max-prob)', color=ps.PCA_EVAR)
    ax2.legend(frameon=False, fontsize=7.5, loc='center right')
    ax.set_title('Train/test loss and peakiness vs epoch')
    # (b) peakiness vs train–test gap, coloured by epoch
    ax = axes[1]
    gap = np.asarray(L['test_loss'], float) - np.asarray(L['train_loss'], float)
    mp = np.asarray(L['maxprob'], float)
    ep = np.asarray(L['epoch'], float)
    sc = ax.scatter(gap, mp, c=ep, cmap='viridis', s=18, zorder=3)
    r = float(np.corrcoef(gap, mp)[0, 1])
    ax.axhline(tgt_mp, color='0.5', ls=':', lw=1, label=f'target ({tgt_mp:.3f})')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label('epoch')
    ax.set_xlabel('train–test gap  (test − train loss)')
    ax.set_ylabel('peakiness (max-prob)')
    ax.set_title(f'Peakiness tracks the gap  (r = {r:.2f})')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ps.label_panels(axes)
    _save(fig, out_dir, 'why_peakier')
    print(f'  WHY: best-test epoch={best}, peakiness there={mp_best:.3f} vs final={L["maxprob"][-1]:.3f} '
          f'(target {tgt_mp:.3f}); peakiness–gap r={r:.2f}')


def _save(fig, out_dir, stem):
    ps.save_fig(fig, out_dir, stem)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bins', type=int, default=91, dest='C')
    ap.add_argument('--n-train', type=int, default=2000)
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--neurons', type=int, default=50)
    ap.add_argument('--tune-width', type=float, default=10.0)
    ap.add_argument('--noise', type=float, default=0.5)
    ap.add_argument('--target-width', type=float, default=9.0)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--log-every', type=int, default=4)
    ap.add_argument('--lam', type=float, default=1.0, help='shape-term weight in PCA+shape')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-root', default='figures/toy_peakiness')
    a = ap.parse_args()
    main(a.C, a.n_train, a.n_test, a.neurons, a.tune_width, a.noise,
         a.target_width, a.hidden, a.epochs, a.log_every, a.lam, a.seed, a.out_root)
