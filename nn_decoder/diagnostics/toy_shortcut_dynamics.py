# -*- coding: utf-8 -*-
"""Toy dynamics: is the PCA decoder taking a SHORTCUT to its loss? (2026-06-03)

Answers Theo's dynamics question: from a uniform start, is the fastest way to cut
the PCA loss to spike at the mode (get the location, ignore the width)? And does
it then get stuck? We instrument the toy training (`toy_peakiness_model` world)
and log, per epoch, where the residual error lives — the LOCATION subspace (top
PCs, 90% var) vs the SHAPE subspace (the rest, unweighted so comparable across
losses) — alongside the decoded peakiness (max-prob).

Two figures under figures/toy_peakiness/:
  shortcut_phaseportrait.png  trajectory in the (location error, peakiness) plane,
                              one path per loss, epoch encoded by colour/spacing.
                              Wide early steps = fast; bunched late = stalled.
  shortcut_decomposition.png  per-loss: location error, shape error, and max-prob
                              vs epoch. Shows location grabbed fast (the shortcut),
                              shape never fixed under PCA, and the stall.

Reading: ALL losses first build a peak to grab the location-dominated loss (the
shared shortcut). KL / flat-L2 then STOP at the target width (shape is penalised);
PCA keeps the location matched but leaves the shape subspace free, so it stalls
peaky — there is no gradient there to broaden it back.

Usage:  python diagnostics/toy_shortcut_dynamics.py
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
import decoder_plotting_utils as dpu  # noqa: E402
from diagnostics.toy_peakiness_model import make_data, MLP, loss_fn  # noqa: E402

LOSS_COLOR = {'PCA (evar-weighted)': '#e6550d', 'flat L2 (Brier)': '#3182bd', 'KL': '#7b3294'}


def _entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -np.sum(p * np.log(p), -1)


def train_logged(Xtr, Ttr, Xte, Tte, kind, pcs, evar, hidden, epochs, log_every, kloc, seed):
    """Train, logging per `log_every` epochs: location-subspace error, shape-
    subspace error (both UNWEIGHTED, over the test set), and decoded max-prob."""
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], hidden, Ttr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    X = torch.tensor(Xtr); T = torch.tensor(Ttr)
    pcs_t = torch.tensor(pcs) if pcs is not None else None
    evar_t = torch.tensor(evar) if evar is not None else None
    Tte_proj = Tte @ pcs.T
    log = {'epoch': [], 'loc': [], 'shape': [], 'maxprob': [], 'H': []}

    def snapshot(ep):
        with torch.no_grad():
            dec = torch.softmax(model(torch.tensor(Xte)), -1).numpy()
        Dp = dec @ pcs.T
        err = ((Dp - Tte_proj) ** 2).mean(0)            # per-PC, unweighted
        log['epoch'].append(ep)
        log['loc'].append(err[:kloc].sum())
        log['shape'].append(err[kloc:].sum())
        log['maxprob'].append(dec.max(1).mean())
        log['H'].append(_entropy(dec).mean())

    snapshot(0)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        pred = torch.softmax(model(X), -1)
        loss_fn(pred, T, kind, pcs_t, evar_t).backward()
        opt.step()
        if ep % log_every == 0:
            snapshot(ep)
    return {k: np.asarray(v) for k, v in log.items()}


def main(C, n_train, n_test, n_neurons, tune_width, noise, target_width,
         hidden, epochs, log_every, seed, out_root):
    dpu.set_style()
    Xtr, Ttr, _ = make_data(n_train, C, n_neurons, tune_width, noise, target_width, seed)
    Xte, Tte, _ = make_data(n_test, C, n_neurons, tune_width, noise, target_width, seed + 1)
    pca = PCA().fit(Ttr)
    pcs = pca.components_.astype(np.float32)
    evar = pca.explained_variance_ratio_.astype(np.float32)
    kloc = int(np.searchsorted(np.cumsum(evar), 0.90)) + 1
    tgt_mp = Tte.max(1).mean()
    print(f'Toy dynamics: location subspace = top {kloc} PCs; target max-prob={tgt_mp:.3f}')

    KINDS = {'PCA (evar-weighted)': ('PCA', pcs, evar),
             'flat L2 (Brier)': ('PCA', pcs, np.full_like(evar, 1.0 / evar.size)),
             'KL': ('KL', None, None)}
    logs = {}
    for name, (kind, pc, ev) in KINDS.items():
        logs[name] = train_logged(Xtr, Ttr, Xte, Tte, kind,
                                   pc if pc is not None else pcs,
                                   ev if ev is not None else evar,
                                   hidden, epochs, log_every, kloc, seed)
        L = logs[name]
        print(f'  {name:20s}: loc {L["loc"][0]:.3f}->{L["loc"][-1]:.3f}  '
              f'shape {L["shape"][0]:.3f}->{L["shape"][-1]:.3f}  '
              f'max-prob {L["maxprob"][0]:.3f}->{L["maxprob"][-1]:.3f}')

    out_dir = Path(out_root)
    _fig_phase(logs, tgt_mp, out_dir)
    _fig_decomp(logs, tgt_mp, out_dir)
    print(f'Done. {out_dir.resolve()}')


def _fig_phase(logs, tgt_mp, out_dir):
    """Phase portrait: location error (x) vs peakiness (y). Each path starts at
    uniform (top-left: high loc error, low peakiness) and is coloured by epoch.
    The shared early dive grabs location; PCA keeps climbing in peakiness while
    KL/flat-L2 stop at the target line."""
    fig, ax = plt.subplots(figsize=(8.2, 6))
    for name, L in logs.items():
        sc = ax.scatter(L['loc'], L['maxprob'], c=L['epoch'], cmap='viridis',
                        s=24, zorder=3)
        ax.plot(L['loc'], L['maxprob'], color=LOSS_COLOR[name], lw=1.3, alpha=0.7,
                zorder=2, label=name)
        ax.scatter(L['loc'][-1], L['maxprob'][-1], color=LOSS_COLOR[name], s=90,
                   edgecolor='k', lw=0.8, zorder=4)
    ax.axhline(tgt_mp, ls='--', color='k', lw=1.2, label=f'target peakiness ({tgt_mp:.3f})')
    ax.annotate('uniform start', xy=(logs['PCA (evar-weighted)']['loc'][0],
                logs['PCA (evar-weighted)']['maxprob'][0]),
                xytext=(10, 20), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.8))
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label('epoch')
    ax.set_xlabel('location-subspace error  (distance to target in the high-evar PCs)')
    ax.set_ylabel('peakiness  (decoded mean max-probability)')
    ax.set_title('The shortcut, as a phase portrait\n'
                 'all losses dive to grab location (→ left) while sharpening (↑); '
                 'KL/flat-L2 stop at the target, PCA overshoots and stalls peaky')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    _save(fig, out_dir, 'shortcut_phaseportrait')


def _fig_decomp(logs, tgt_mp, out_dir):
    """Per loss: location error, shape error, and peakiness vs epoch. The
    location error is grabbed fast (the shortcut); under PCA the shape error
    never falls and max-prob keeps climbing; under KL/flat-L2 both errors fall
    and peakiness halts at target."""
    n = len(logs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), sharey=False)
    for ax, (name, L) in zip(axes, logs.items()):
        ax.plot(L['epoch'], L['loc'], color='#1b9e77', lw=2.2, label='location error')
        ax.plot(L['epoch'], L['shape'], color='#d95f02', lw=2.2, label='shape error')
        ax.set_yscale('log'); ax.set_xlabel('epoch')
        ax.set_title(name, color=LOSS_COLOR[name], fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('subspace error (unweighted, log)')
        ax2 = ax.twinx()
        ax2.plot(L['epoch'], L['maxprob'], color='#7570b3', lw=2.0, ls='--')
        ax2.axhline(tgt_mp, color='0.5', ls=':', lw=1)
        ax2.set_ylabel('max-prob (dashed purple)', color='#7570b3', fontsize=9)
        ax2.set_ylim(0, max(0.3, max(L['maxprob']) * 1.1))
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8, loc='center right')
    fig.suptitle('Where the loss reduction comes from — location grabbed fast, '
                 'shape only fixed by KL/flat-L2 (under PCA it never falls → stuck peaky)',
                 y=1.03, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'shortcut_decomposition')


def _save(fig, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'{stem}.{ext}', bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(f'  -> {stem}.png/.svg')


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
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-root', default='figures/toy_peakiness')
    a = ap.parse_args()
    main(a.C, a.n_train, a.n_test, a.neurons, a.tune_width, a.noise,
         a.target_width, a.hidden, a.epochs, a.log_every, a.seed, a.out_root)
