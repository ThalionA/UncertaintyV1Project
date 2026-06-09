# -*- coding: utf-8 -*-
"""Toy model: prove the PCA-loss peakiness mechanism under full control (2026-06-03).

Fully synthetic, no real data. A decoder maps a noisy population code to a
posterior over a 1-D circular "orientation". The targets are BROAD bumps of FIXED
width whose location varies; the input determines location only up to noise (the
irreducible location uncertainty). We train the same small MLP under three losses
— evar-weighted L2 (= the PCA loss), flat L2 (Brier), forward KL — and ask what
each produces.

This isolates the loss from every real-data confound and lets us test the
competing accounts of why the PCA loss goes peaky:

  * SPECTRAL test (the decisive one): decompose the decoded-vs-target error by
    principal component. If the peakiness lives in the loss-BLIND subspace, the
    PCA-loss decoder should match the target on the high-evar (location) PCs just
    as well as KL, but diverge badly on the low-evar (width/shape) PCs — exactly
    the components its loss down-weights to ~0 — while KL stays matched there.
  * CAPACITY / NOISE sweeps: does the over-sharpening grow with hidden width
    (overfitting fills the free subspace) and with input noise (more location
    uncertainty)?

Outputs (PNG+SVG) under figures/toy_peakiness/:
  toy_examples.png      example decoded posteriors per loss vs target
  toy_spectrum.png      per-PC decoded-vs-target error + evar weighting (the proof)
  toy_sweeps.png        decoded entropy gap vs hidden width and vs input noise

Usage
-----
    python diagnostics/toy_peakiness_model.py
    python diagnostics/toy_peakiness_model.py --neurons 60 --hidden 32 --noise 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402


# ----------------------------------------------------------------------
# Synthetic world
# ----------------------------------------------------------------------

def circular_bump(centre, C, width, rng=None):
    """Normalised wrapped-Gaussian bump over C bins at `centre` with sd `width`."""
    x = np.arange(C)
    d = np.minimum(np.abs(x - centre), C - np.abs(x - centre))  # circular distance
    p = np.exp(-0.5 * (d / width) ** 2)
    return p / p.sum()


def make_data(n, C, n_neurons, tune_width, noise, target_width, seed):
    """Returns (X, targets, locs). X: (n, n_neurons) noisy population code;
    targets: (n, C) broad bumps of FIXED width at the (hidden) location."""
    rng = np.random.default_rng(seed)
    locs = rng.uniform(0, C, size=n)
    prefs = np.linspace(0, C, n_neurons, endpoint=False)
    # circular tuning of each neuron to the location
    dd = np.minimum(np.abs(locs[:, None] - prefs[None, :]),
                    C - np.abs(locs[:, None] - prefs[None, :]))
    rates = np.exp(-0.5 * (dd / tune_width) ** 2)
    X = rates + rng.normal(0, noise, size=rates.shape)          # location uncertainty
    targets = np.stack([circular_bump(s, C, target_width) for s in locs])
    return (X.astype(np.float32), targets.astype(np.float32), locs)


# ----------------------------------------------------------------------
# Model + losses
# ----------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))   # raw logits


def loss_fn(pred, target, kind, pcs=None, evar=None):
    """pred, target: (B, C). Mirrors the real losses."""
    if kind == 'KL':
        return torch.sum(target * (torch.log(target + 1e-12) - torch.log(pred + 1e-12)), -1).mean()
    proj_p = pred @ pcs.T
    proj_t = target @ pcs.T
    return (evar * (proj_p - proj_t) ** 2).sum(-1).mean() * 100.0


def train(Xtr, Ttr, kind, pcs, evar, d_hidden, epochs, seed):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], d_hidden, Ttr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    X = torch.tensor(Xtr); T = torch.tensor(Ttr)
    pcs_t = torch.tensor(pcs) if pcs is not None else None
    evar_t = torch.tensor(evar) if evar is not None else None
    for _ in range(epochs):
        opt.zero_grad()
        pred = torch.softmax(model(X), -1)
        loss_fn(pred, T, kind, pcs_t, evar_t).backward()
        opt.step()
    return model


def decode(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), -1).numpy()


def _entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -np.sum(p * np.log(p), -1)


# ----------------------------------------------------------------------

def main(C, n_train, n_test, n_neurons, tune_width, noise, target_width,
         hidden, epochs, seed, out_root):
    ps.apply()
    Xtr, Ttr, _ = make_data(n_train, C, n_neurons, tune_width, noise, target_width, seed)
    Xte, Tte, _ = make_data(n_test, C, n_neurons, tune_width, noise, target_width, seed + 1)
    # PCA basis on all training targets (all_trials), like the real pipeline
    pca = PCA().fit(Ttr)
    pcs = pca.components_.astype(np.float32)
    evar = pca.explained_variance_ratio_.astype(np.float32)
    Htgt = _entropy(Tte).mean()

    LOSSES = ['PCA (evar-weighted)', 'flat L2 (Brier)', 'KL']
    KINDS = {'PCA (evar-weighted)': ('PCA', pcs, evar),
             'flat L2 (Brier)': ('PCA', pcs, np.full_like(evar, 1.0 / evar.size)),
             'KL': ('KL', None, None)}
    decoded = {}
    print(f'Toy world: C={C}, neurons={n_neurons}, noise={noise}, '
          f'target_width={target_width}, hidden={hidden}; target H={Htgt:.2f}')
    for name in LOSSES:
        kind, pc, ev = KINDS[name]
        model = train(Xtr, Ttr, kind, pc, ev, hidden, epochs, seed)
        dec = decode(model, Xte)
        decoded[name] = dec
        print(f'  {name:20s}: decoded H={_entropy(dec).mean():.2f}  '
              f'max-prob={dec.max(1).mean():.3f}  (target H={Htgt:.2f}, '
              f'mp={Tte.max(1).mean():.3f})')

    out_dir = Path(out_root)
    _fig_examples(decoded, Tte, out_dir, Htgt)
    _fig_spectrum(decoded, Tte, pcs, evar, out_dir)
    _fig_sweeps(C, n_train, n_test, n_neurons, tune_width, target_width, epochs,
                seed, out_dir)
    print(f'Done. {out_dir.resolve()}')


def _fig_examples(decoded, Tte, out_dir, Htgt):
    rng = np.random.default_rng(0)
    picks = rng.choice(Tte.shape[0], 4, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=ps.figsize(4, 1), sharey=True)
    x = np.arange(Tte.shape[1])
    for ax, tr in zip(axes, picks):
        ps.target_band(ax, x, Tte[tr], label='target')
        for name, dec in decoded.items():
            ax.plot(x, dec[tr], color=ps.color(name), lw=1.8, label=name)
        ax.set_xlabel('orientation bin')
        ax.set_title(f'trial {tr}', fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, loc='best')
    axes[0].set_ylabel('probability')
    ps.label_panels(axes)
    fig.suptitle('Toy model: decoded posteriors', y=1.02)
    _save(fig, out_dir, 'toy_examples')


def _fig_spectrum(decoded, Tte, pcs, evar, out_dir):
    """THE PROOF: per-PC mean squared decoded-vs-target error, with the evar
    weighting overlaid. PCA matches the high-evar (location) PCs like KL but
    diverges on the low-evar (width/shape) PCs it doesn't weight; KL stays
    matched there. The peakiness lives in the loss-blind subspace."""
    Tt = Tte @ pcs.T
    fig, ax = plt.subplots(figsize=ps.figsize(2, 1))
    k = np.arange(len(evar))
    for name, dec in decoded.items():
        Dp = dec @ pcs.T
        err = ((Dp - Tt) ** 2).mean(0)
        ax.semilogy(k, err + 1e-12, color=ps.color(name), lw=2.0, marker='o',
                    ms=2.5, label=name)
    ax.set_xlabel('principal component k   (low k = location, high k = width/shape)')
    ax.set_ylabel('mean squared decoded−target projection error')
    ax.legend(frameon=False, fontsize=9, loc='best')
    ax2 = ax.twinx()
    ax2.semilogy(k, evar + 1e-12, color='0.5', ls='--', lw=1.2)
    ax2.set_ylabel('evar_k  (PCA loss weight, dashed grey)', color='0.4')
    ax.set_title('Per-PC decoded−target error spectrum')
    _save(fig, out_dir, 'toy_spectrum')


def _fig_sweeps(C, n_train, n_test, n_neurons, tune_width, target_width, epochs,
                seed, out_dir):
    """Entropy gap (target − decoded; positive = peakier than target) vs hidden
    width and vs input noise, per loss. Tests the overfitting / uncertainty
    contributions."""
    fig, axes = plt.subplots(1, 2, figsize=ps.figsize(2, 1))
    LOSSES = ['PCA (evar-weighted)', 'flat L2 (Brier)', 'KL']

    def run(hidden, noise):
        Xtr, Ttr, _ = make_data(n_train, C, n_neurons, tune_width, noise, target_width, seed)
        Xte, Tte, _ = make_data(n_test, C, n_neurons, tune_width, noise, target_width, seed + 1)
        pca = PCA().fit(Ttr); pcs = pca.components_.astype(np.float32)
        evar = pca.explained_variance_ratio_.astype(np.float32)
        Htgt = _entropy(Tte).mean()
        gaps = {}
        for name in LOSSES:
            if name == 'KL':
                kind, pc, ev = 'KL', None, None
            elif name == 'flat L2 (Brier)':
                kind, pc, ev = 'PCA', pcs, np.full_like(evar, 1.0 / evar.size)
            else:
                kind, pc, ev = 'PCA', pcs, evar
            m = train(Xtr, Ttr, kind, pc, ev, hidden, epochs, seed)
            gaps[name] = Htgt - _entropy(decode(m, Xte)).mean()  # >0 = peakier
        return gaps

    hiddens = [4, 8, 16, 32, 64]
    g_h = [run(h, noise=0.5) for h in hiddens]
    for name in LOSSES:
        axes[0].plot(hiddens, [g[name] for g in g_h], color=ps.color(name),
                     lw=2, marker='o', label=name)
    axes[0].axhline(0, color='k', lw=0.6)
    axes[0].set_xlabel('hidden width'); axes[0].set_ylabel('entropy gap  (target − decoded)')
    axes[0].set_title('vs capacity  (noise=0.5)'); axes[0].legend(frameon=False, fontsize=8, loc='best')

    noises = [0.1, 0.3, 0.5, 0.8, 1.2]
    g_n = [run(hidden=32, noise=nz) for nz in noises]
    for name in LOSSES:
        axes[1].plot(noises, [g[name] for g in g_n], color=ps.color(name),
                     lw=2, marker='o', label=name)
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set_xlabel('input noise (location uncertainty)')
    axes[1].set_ylabel('entropy gap  (target − decoded)')
    axes[1].set_title('vs location uncertainty  (hidden=32)')
    ps.label_panels(axes)
    fig.suptitle('Toy model: decoded entropy gap', y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, 'toy_sweeps')


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
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-root', default='figures/toy_peakiness')
    a = ap.parse_args()
    main(a.C, a.n_train, a.n_test, a.neurons, a.tune_width, a.noise,
         a.target_width, a.hidden, a.epochs, a.seed, a.out_root)
