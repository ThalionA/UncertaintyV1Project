# -*- coding: utf-8 -*-
"""Visualise the training trajectory of the posteriors + the loss landscape that
drives PCA toward peaky outputs (2026-06-03, Theo's request 3).

Two figures, on the existing `loss_comparison_v1` data (no re-run):

1. POSTERIOR MORPHING — for a few example test trials, re-decode at each weight
   snapshot and draw the decoded posterior coloured by epoch, with the IO target
   overlaid. Columns compare PCA (peaky) vs a calibrated loss (KL). You see the
   PCA posterior start broad (≈uniform) and progressively spike, sailing past the
   target width, while KL settles onto the target.
   -> figures/.../trajectory_landscape/morph_<target>_m<mouse>.png

2. LOSS LANDSCAPE ALONG THE WIDTH AXIS — the "what drives them" plot. For every
   test trial we take its IO target and walk a one-parameter sharpen/broaden
   family p_γ ∝ target^γ (γ<1 broader, γ=1 = target, γ>1 sharper), then evaluate
   each loss(p_γ, target) and average over trials. All proper losses are
   minimised at γ=1 (pred=target). The diagnostic is the *curvature*: PCA's
   evar-weighted basin along the sharpening axis is far SHALLOWER than the
   unweighted-L2 / KL / JS basins — a weak restoring force — because sharpening
   moves mostly trailing (width) PCs that the evar weighting discards. Flat-evar
   PCA (uniform weights) is shown as the control: its basin matches plain L2.
   -> figures/.../trajectory_landscape/loss_vs_width_<target>_m<mouse>.png

Usage
-----
    python diagnostics/posterior_trajectory_landscape.py
    python diagnostics/posterior_trajectory_landscape.py --mouse 0 --calibrated KL
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
from nn_classifier import SimpleFlexibleNNClassifier, fit_loss_per_trial  # noqa: E402

KEY_WIN, KEY_WOUT = 'layers.0.weight', 'layers.1.weight'

# human-readable decoder names for titles/labels (never the cryptic arch codes)
ARCH_NAME = {'spat': 'spatial', 'temp': 'temporal'}


def _slug(target, loss, window, bin_ms):
    return f'{target}_{loss}_{window}_{bin_ms}ms' + ('_all' if loss == 'PCA' else '')


def _as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def _entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _normalize(p):
    p = np.clip(np.asarray(p, float), 0, None)
    return p / p.sum(axis=-1, keepdims=True)


def _sharpen(p, gamma):
    """p_γ ∝ p^γ, renormalised. γ>1 sharpens, γ<1 broadens, γ=1 identity."""
    return _normalize(np.asarray(p, float) ** gamma)


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
    xb = torch.tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        if arch == 'spat':
            probs = torch.softmax(model(xb.mean(dim=1)), dim=-1)
        else:
            probs = torch.softmax(model(xb), dim=-1).mean(dim=1)
    return probs.numpy()


def load_checkpoint(results_root, run_name, target, loss, window, bin_ms, split, mouse, arch):
    pt = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / \
        'checkpoints' / f'mouse_{mouse}_{split}.pt'
    if not pt.is_file():
        return None
    ck = torch.load(str(pt), map_location='cpu', weights_only=False)
    if arch not in ck:
        return None
    return ck[arch]


def load_mat(results_root, run_name, target, loss, window, bin_ms, split, mouse):
    mat = Path(results_root) / run_name / _slug(target, loss, window, bin_ms) / f'{split}.mat'
    d = sio.loadmat(str(mat), simplify_cells=True)
    return d['results'][f'mouse_{mouse}'], d['config']


# ----------------------------------------------------------------------

def fig_morph(results_root, run_name, target, window, bin_ms, split, mouse,
              arch, calibrated, activation, out_dir, info):
    """Decoded posterior vs training epoch for a few example trials: PCA vs
    a calibrated loss. Lines coloured by snapshot epoch; target overlaid."""
    losses = ['PCA', calibrated]
    subs = {l: load_checkpoint(results_root, run_name, target, l, window, bin_ms, split, mouse, arch)
            for l in losses}
    subs = {l: s for l, s in subs.items() if s is not None}
    if 'PCA' not in subs:
        print('  [morph] no PCA checkpoint'); return
    # choose example trials by target width (narrow / mid / broad)
    m_res, _ = load_mat(results_root, run_name, target, 'PCA', window, bin_ms, split, mouse)
    targets = np.asarray(m_res['Dist'][arch]['target'], float)
    H = _entropy(targets)
    order = np.argsort(H)
    picks = [order[int(0.15 * len(order))], order[int(0.5 * len(order))],
             order[int(0.85 * len(order))]]
    nrow, ncol = len(picks), len(subs)
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow, mw=1.3),
                             squeeze=False, sharex=True)
    cmap = plt.get_cmap('viridis')
    x = np.arange(targets.shape[1])
    for c, (loss, sub) in enumerate(subs.items()):
        hist = sub['history']
        eps = np.asarray(hist['snapshot_epochs'], int)
        sds = hist['state_dicts']
        X = sub['X_test']
        # decode all snapshots once
        decoded_by_ep = [(_decode(_build_model(sd, activation), X, arch)) for sd in sds]
        for r, tr in enumerate(picks):
            ax = axes[r][c]
            for k, dec in enumerate(decoded_by_ep):
                ax.plot(x, dec[tr], color=cmap(k / max(1, len(sds) - 1)), lw=1.3,
                        alpha=0.9)
            ax.plot(x, targets[tr], color='k', lw=2.0, ls='--', label='IO target')
            if r == 0:
                ax.set_title(f'{loss}', color=ps.color(loss), fontsize=12)
            if c == 0:
                ax.set_ylabel(f'trial {tr}', fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel('orientation bin')
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7, loc='upper right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(eps.min(), eps.max()))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label('epoch (dark → bright)')
    ps.label_panels(axes)
    fig.suptitle(f'Posterior morphing across training — {ARCH_NAME.get(arch, arch)} decoder',
                 y=1.02)
    _save(fig, out_dir, f'morph_{target}_m{mouse}')


def fig_loss_landscape(results_root, run_name, target, window, bin_ms, split,
                       mouse, arch, activation, out_dir, info):
    """Average loss(p_γ, target) along the sharpen/broaden axis p_γ ∝ target^γ,
    per loss — the driver. PCA's evar-weighted basin is shallow on the sharp
    side; flat-evar / KL / JS are steep (strong restoring force)."""
    m_res, _ = load_mat(results_root, run_name, target, 'PCA', window, bin_ms, split, mouse)
    Dist = m_res['Dist']
    targets = np.asarray(Dist[arch]['target'], float)
    pcs = torch.tensor(np.asarray(Dist['pcs'], float))      # pcs/evar are Dist-level
    evar = torch.tensor(np.asarray(Dist['explained_var'], float))
    evar_flat = torch.full_like(evar, 1.0 / evar.numel())
    T = torch.tensor(targets)

    gammas = np.geomspace(0.35, 4.0, 31)
    loss_specs = [('PCA', 'PCA', pcs, evar), ('PCA-flat', 'PCA', pcs, evar_flat),
                  ('CE', 'CE', None, None), ('KL', 'KL', None, None),
                  ('JS', 'JS', None, None), ('Wasserstein', 'Wasserstein', None, None)]
    curves = {}
    for name, key, pc, ev in loss_specs:
        vals = []
        for g in gammas:
            pred = torch.tensor(_sharpen(targets, g))
            per = fit_loss_per_trial(pred, T, key, pc, ev)
            vals.append(float(per.mean()))
        curves[name] = np.asarray(vals)

    # network's actual operating sharpness: γ matching the PCA-decoded entropy
    dec_pca = np.asarray(Dist[arch]['decoded'], float)
    Hdec = float(np.mean(_entropy(dec_pca)))
    Hgam = np.array([np.mean(_entropy(_sharpen(targets, g))) for g in gammas])
    g_net = float(gammas[np.argmin(np.abs(Hgam - Hdec))])

    # The driver: asymmetry of the basin around γ=1. ratio = (cost of sharpening
    # to γ=2) / (cost of broadening to γ=0.5). <1 => sharpening is cheaper than
    # broadening => the loss tolerates / pulls toward peaky; >1 => restoring force
    # against sharpening (stays calibrated). All losses have their MIN at γ=1.
    gi1 = int(np.argmin(np.abs(gammas - 1.0)))
    gi2 = int(np.argmin(np.abs(gammas - 2.0)))
    gi05 = int(np.argmin(np.abs(gammas - 0.5)))
    ratios = {}
    for name, y in curves.items():
        sharp, broad = y[gi2] - y[gi1], y[gi05] - y[gi1]
        ratios[name] = sharp / broad if abs(broad) > 1e-12 else float('nan')

    fig, axes = plt.subplots(2, 3, figsize=ps.figsize(3, 2), squeeze=False)
    for i, (ax, (name, _, _, _)) in enumerate(zip(axes.ravel(), loss_specs)):
        y = curves[name]
        ax.plot(gammas, y, color=ps.color(name), lw=2.4)
        ax.axvline(1.0, ls='--', color='k', lw=1,
                   label='γ=1 (IO target)' if i == 0 else None)
        ax.axvline(g_net, ls=':', color=ps.PCA_EVAR, lw=1.5,
                   label=f'PCA peakiness (γ≈{g_net:.1f})' if i == 0 else None)
        r = ratios[name]
        ax.set_title(f'{name}   (sharp/broad = {r:.2f})',
                     color=ps.color(name), fontsize=11)
        ax.set_xscale('log')
        ax.set_xlabel('γ   (broaden ← 1 → sharpen)')
        if i % 3 == 0:
            ax.set_ylabel('mean loss(p_γ, target)')
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc='best')
    ps.label_panels(axes)
    fig.suptitle(f'Loss along the sharpen/broaden axis — {ARCH_NAME.get(arch, arch)} decoder',
                 y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f'loss_vs_width_{target}_m{mouse}')

    print('  sharp-side vs broad-side cost (Δloss γ1→γ2  /  Δloss γ1→γ0.5):')
    for name in curves:
        y = curves[name]
        print(f'     {name:12s} sharp={y[gi2]-y[gi1]:+.4g} '
              f'broad={y[gi05]-y[gi1]:+.4g} ratio={ratios[name]:.2f}')
    return g_net


def _save(fig, out_dir, stem):
    ps.save_fig(fig, out_dir, stem)


def main(run_name, target, window, bin_ms, split, mouse, arch, calibrated,
         results_root, out_root):
    ps.apply()
    cfg_mat = Path(results_root) / run_name / _slug(target, 'KL', window, bin_ms) / f'{split}.mat'
    activation = 'tanh'
    if cfg_mat.is_file():
        activation = sio.loadmat(str(cfg_mat), simplify_cells=True)['config'].get(
            'activation_function', 'tanh')
    info = f'{target} {window} {bin_ms}ms {split} mouse {mouse}'
    out_dir = Path(out_root) / run_name / 'trajectory_landscape'
    print(f'Trajectory + landscape: {info} | arch={arch} | activation={activation}')
    fig_morph(results_root, run_name, target, window, bin_ms, split, mouse,
              arch, calibrated, activation, out_dir, info)
    fig_loss_landscape(results_root, run_name, target, window, bin_ms, split,
                       mouse, arch, activation, out_dir, info)
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', default='loss_comparison_v1')
    ap.add_argument('--target', default='Q')
    ap.add_argument('--window', default='half')
    ap.add_argument('--bin', type=int, default=100, dest='bin_ms')
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--mouse', type=int, default=0)
    ap.add_argument('--arch', default='spat', choices=('spat', 'temp'))
    ap.add_argument('--calibrated', default='KL', help='loss to contrast against PCA in morph')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_sweep_plots')
    a = ap.parse_args()
    main(a.run_name, a.target, a.window, a.bin_ms, a.split, a.mouse, a.arch,
         a.calibrated, a.results_root, a.out_root)
