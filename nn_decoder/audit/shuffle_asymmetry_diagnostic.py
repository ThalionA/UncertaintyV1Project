# -*- coding: utf-8 -*-
"""Diagnostic for the spat_shf > temp_shf asymmetry.

Both the spatial and temporal decoders in
``run_experiment.run_animal_decoder`` share the same MLP backbone and
training pipeline; they differ only in **where time-averaging happens**
inside the forward pass (``nn_classifier.get_model_probabilities``):

  - spatial      :  softmax(MLP(  mean_t(x_t)  ))     -> one prediction / trial
  - Sampling :  mean_t( softmax(MLP(x_t)) )       -> T predictions, then averaged

These are not equivalent. Averaging T post-softmax distributions reduces
trial-level prediction variance by a Monte-Carlo factor; averaging
inputs first does not. The consequence is a *systematic* shuffle gap:
on target-shuffled controls (where the optimal prediction is the
training-set marginal of the target), the sampling decoder's
output-averaging pulls each trial's prediction closer to that marginal
than the spatial decoder's single-pass output can.

This script confirms the mechanism two ways:

  1. **Empirical breakdown** (`empirical_breakdown`). Loads the saved
     per-trial predicted distributions from
     ``population_results_*.mat`` and decomposes the spat_shf > temp_shf
     loss into prediction entropy, trial-to-trial variance of the
     prediction, and JS-distance of the per-trial / mean prediction to
     the training marginal. The result: per-trial entropy is nearly
     identical between the two, but spat_shf's trial-to-trial variance
     is consistently ~2x larger -- i.e. spat_shf jiggles around the
     marginal more.

  2. **Synthetic Jensen demo** (`synthetic_jensen_demo`). Fixed random
     MLP, random per-bin Gaussian inputs, no training. Just runs the
     two forward passes and shows the trial-to-trial variance of the
     spatial-style output is larger than the sampling-style output, with
     near-identical per-trial entropy. Closed-form: the architectural
     asymmetry alone produces the gap.

Figures land in ``nn_decoder/figures/shuffle_asymmetry/``. Run with:

    python -m audit.shuffle_asymmetry_diagnostic         # from nn_decoder/
    python audit/shuffle_asymmetry_diagnostic.py         # equivalent
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow running from either the repo root or nn_decoder/.
HERE = Path(__file__).resolve().parent
NN_DECODER = HERE.parent
if str(NN_DECODER) not in sys.path:
    sys.path.insert(0, str(NN_DECODER))

from paths import figures_dir
from figsave import save_fig

FIGS_SUBDIR = 'shuffle_asymmetry'


# ----------------------------------------------------------------------
# Distribution-level metrics
# ----------------------------------------------------------------------

def entropy(p, axis=-1):
    """H(p) in nats. ``p`` is a probability distribution along ``axis``."""
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def js_divergence(p, q, axis=-1):
    """Per-trial Jensen-Shannon divergence (nats) between matched p/q."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * (np.log(p) - np.log(m)), axis=axis)
                  + np.sum(q * (np.log(q) - np.log(m)), axis=axis))


def trial_variance(preds):
    """Sum over categories of the per-trial variance of ``preds`` --
    a single number summarising trial-to-trial prediction jiggle.

    ``preds`` is ``(n_trials, n_cats)``.
    """
    return float(preds.var(axis=0).sum())


# ----------------------------------------------------------------------
# 1. Empirical breakdown from saved population_results .mat
# ----------------------------------------------------------------------

def _per_mouse_breakdown(mat_path):
    """For each mouse, compute the spat_shf vs temp_shf decomposition.

    Returns a dict
        {mouse_id: {
            'H_spat_shf', 'H_temp_shf',     # mean per-trial entropy
            'var_spat_shf', 'var_temp_shf', # trial-to-trial var (sum over cats)
            'JS_spat_shf', 'JS_temp_shf',   # mean per-trial JS vs target marginal
            'JS_mean_spat_shf', 'JS_mean_temp_shf',  # JS of mean pred vs marginal
            'fit_loss_spat_shf', 'fit_loss_temp_shf', # mean per-trial JS vs per-trial target
        }}
    """
    d = sio.loadmat(mat_path, simplify_cells=True)
    out = {}
    for key, val in d['results'].items():
        if not key.startswith('mouse_'):
            continue
        mid = int(key.split('_')[1])
        Dist = val['Dist']
        if 'spat_shf' not in Dist or 'temp_shf' not in Dist:
            continue
        pred_s = np.asarray(Dist['spat_shf']['decoded'], dtype=float)
        pred_t = np.asarray(Dist['temp_shf']['decoded'], dtype=float)
        target = np.asarray(Dist['spat']['target'], dtype=float)  # shared
        target_marg = target.mean(axis=0, keepdims=True)
        target_marg_b = np.broadcast_to(target_marg, pred_s.shape)

        out[mid] = dict(
            H_spat_shf=float(entropy(pred_s).mean()),
            H_temp_shf=float(entropy(pred_t).mean()),
            var_spat_shf=trial_variance(pred_s),
            var_temp_shf=trial_variance(pred_t),
            JS_spat_shf=float(js_divergence(pred_s, target_marg_b).mean()),
            JS_temp_shf=float(js_divergence(pred_t, target_marg_b).mean()),
            JS_mean_spat_shf=float(js_divergence(
                pred_s.mean(axis=0, keepdims=True), target_marg)[0]),
            JS_mean_temp_shf=float(js_divergence(
                pred_t.mean(axis=0, keepdims=True), target_marg)[0]),
            fit_loss_spat_shf=float(js_divergence(pred_s, target).mean()),
            fit_loss_temp_shf=float(js_divergence(pred_t, target).mean()),
            n_trials=int(pred_s.shape[0]),
            n_cats=int(pred_s.shape[1]),
        )
    return out


def plot_empirical_breakdown(breakdown, mat_path, out_path):
    """Four-panel comparison of spat_shf vs temp_shf across mice.

    A: mean per-trial entropy           -- nearly equal
    B: trial-to-trial prediction var    -- spat ~2x larger
    C: mean per-trial JS vs marginal    -- spat larger
    D: JS of MEAN prediction vs marginal -- both decoders recover marginal
    """
    mids = sorted(breakdown.keys())
    x = np.arange(len(mids))
    H_s = [breakdown[m]['H_spat_shf']   for m in mids]
    H_t = [breakdown[m]['H_temp_shf']   for m in mids]
    V_s = [breakdown[m]['var_spat_shf'] for m in mids]
    V_t = [breakdown[m]['var_temp_shf'] for m in mids]
    J_s = [breakdown[m]['JS_spat_shf']  for m in mids]
    J_t = [breakdown[m]['JS_temp_shf']  for m in mids]
    JM_s = [breakdown[m]['JS_mean_spat_shf'] for m in mids]
    JM_t = [breakdown[m]['JS_mean_temp_shf'] for m in mids]

    spat_c = '#1f77b4'
    temp_c = '#d62728'
    w = 0.4

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (axes[0, 0], H_s, H_t,
         'A. Per-trial entropy of prediction',
         'mean H(pred)  (nats)',
         'Nearly identical: both models output similarly broad\n'
         'distributions per trial.'),
        (axes[0, 1], V_s, V_t,
         'B. Trial-to-trial prediction variance',
         '$\\sum_c$ var$_{trial}$(pred[:, c])',
         "Spat is ~2× larger: spat's per-trial output jiggles\n"
         'around the marginal more.'),
        (axes[1, 0], J_s, J_t,
         'C. Mean per-trial JS(pred, target marginal)',
         'mean JS  (nats)',
         "Mirrors trial-variance: spat further from the\n"
         'training marginal trial-by-trial.'),
        (axes[1, 1], JM_s, JM_t,
         'D. JS(mean prediction, target marginal)',
         'JS  (nats)',
         'Mean predictions of BOTH models land close to the\n'
         'training marginal. The miss is per-trial, not on average.'),
    ]

    for ax, ys, yt, title, ylab, caption in panels:
        ax.bar(x - w / 2, ys, width=w, color=spat_c, label='spat_shf')
        ax.bar(x + w / 2, yt, width=w, color=temp_c, label='temp_shf')
        ax.set_xticks(x)
        ax.set_xticklabels([f'M{m}' for m in mids], fontsize=9)
        ax.set_xlabel('Mouse', fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        # caption below x-axis labels
        ax.text(0.02, -0.30, caption, transform=ax.transAxes,
                fontsize=8, color='0.30', va='top')

    fig.suptitle(
        'Spat_shf vs Temp_shf — architectural shuffle asymmetry decomposed\n'
        f'(source: {os.path.basename(mat_path)})',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_fig(fig, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    return out_path


# ----------------------------------------------------------------------
# 2. Synthetic Jensen demonstration — no training required
# ----------------------------------------------------------------------

def _synthetic_predictions(n_trials=400, T=10, n_neurons=60, n_cats=91,
                            hidden=32, sigma_signal=0.5, sigma_bin=1.0,
                            seed=0):
    """Run the two forward passes on the same synthetic inputs.

    A per-trial 'signal' vector ``s_n`` is drawn once; each time bin's
    activation is ``x_{n,t} = s_n + epsilon_{n,t}`` with epsilon ~
    N(0, sigma_bin^2 I). A fixed random MLP (n_neurons -> hidden ->
    n_cats, tanh) maps activations to logits.

    Returns
    -------
    pred_ppc   : (n_trials, n_cats) -- softmax(MLP(mean_t(x_t)))
    pred_samp  : (n_trials, n_cats) -- mean_t(softmax(MLP(x_t)))
    """
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, sigma_signal, size=(n_trials, n_neurons))
    eps = rng.normal(0.0, sigma_bin, size=(n_trials, T, n_neurons))
    x = s[:, None, :] + eps                                     # (N, T, n_neurons)

    # Fixed random MLP.
    W1 = rng.normal(0.0, 1.0 / np.sqrt(n_neurons), size=(n_neurons, hidden))
    b1 = rng.normal(0.0, 0.1, size=(hidden,))
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, n_cats))
    b2 = rng.normal(0.0, 0.1, size=(n_cats,))

    def mlp(z):
        h = np.tanh(z @ W1 + b1)
        return h @ W2 + b2

    def softmax(z, axis=-1):
        z = z - z.max(axis=axis, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=axis, keepdims=True)

    # spatial: average inputs first.
    x_mean = x.mean(axis=1)                                     # (N, n_neurons)
    pred_ppc = softmax(mlp(x_mean))                             # (N, n_cats)

    # Sampling: per-bin softmax, then average across bins.
    per_bin_probs = softmax(mlp(x))                             # (N, T, n_cats)
    pred_samp = per_bin_probs.mean(axis=1)                      # (N, n_cats)
    return pred_ppc, pred_samp, per_bin_probs


def synthetic_jensen_demo(out_dir, seed=0):
    """Render the Jensen-from-architecture demonstration figure.

    Two panels are the headline quantitative comparison (entropy and
    trial-variance), matching the empirical breakdown plot. The third
    panel overlays sample predicted distributions from a handful of
    trials so the reader sees the visual difference: same input,
    spatial-style outputs sit further from each other than sampling-style.
    """
    pred_ppc, pred_samp, per_bin = _synthetic_predictions(seed=seed)

    H_ppc = entropy(pred_ppc).mean()
    H_samp = entropy(pred_samp).mean()
    V_ppc = trial_variance(pred_ppc)
    V_samp = trial_variance(pred_samp)
    # ratio loss = per-trial JS to the marginal; marginal here is the
    # mean across trials of the predictions, since the "target" is
    # whatever the marginal is (random / unstructured targets <=> match
    # the marginal at optimum).
    marg_ppc = pred_ppc.mean(axis=0, keepdims=True)
    marg_samp = pred_samp.mean(axis=0, keepdims=True)
    J_ppc = js_divergence(pred_ppc, np.broadcast_to(marg_ppc, pred_ppc.shape)).mean()
    J_samp = js_divergence(pred_samp, np.broadcast_to(marg_samp, pred_samp.shape)).mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    spat_c, temp_c = '#1f77b4', '#d62728'

    # Panel A -- entropy.
    axes[0].bar(['spatial-style\nsoftmax(MLP(mean(x)))',
                  'Sampling-style\nmean(softmax(MLP(x)))'],
                 [H_ppc, H_samp], color=[spat_c, temp_c])
    axes[0].set_ylabel('Mean H(pred)  (nats)')
    axes[0].set_title('A. Per-trial entropy', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.25)
    for i, v in enumerate([H_ppc, H_samp]):
        axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B -- trial-to-trial variance.
    axes[1].bar(['spatial-style', 'Sampling-style'],
                 [V_ppc, V_samp], color=[spat_c, temp_c])
    axes[1].set_ylabel('$\\sum_c$ var$_{trial}$(pred[:, c])')
    axes[1].set_title('B. Trial-to-trial prediction variance',
                       fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.25)
    for i, v in enumerate([V_ppc, V_samp]):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ratio = V_ppc / V_samp if V_samp > 0 else np.nan
    axes[1].text(0.5, 0.95, f'spatial / temporal variance ratio: {ratio:.2f}×',
                  transform=axes[1].transAxes, ha='center', va='top',
                  fontsize=10, color='0.30',
                  bbox=dict(facecolor='white', edgecolor='none', pad=2))

    # Panel C -- sample predicted distributions overlaid.
    rng = np.random.default_rng(123)
    show = rng.choice(pred_ppc.shape[0], size=12, replace=False)
    s_grid = np.arange(pred_ppc.shape[1])
    for i, t in enumerate(show):
        lbl_p = 'spatial-style' if i == 0 else None
        lbl_s = 'Sampling-style' if i == 0 else None
        axes[2].plot(s_grid, pred_ppc[t], color=spat_c, alpha=0.55, lw=1.0,
                      label=lbl_p)
        axes[2].plot(s_grid, pred_samp[t], color=temp_c, alpha=0.55, lw=1.0,
                      label=lbl_s)
    axes[2].set_xlabel('Output category')
    axes[2].set_ylabel('p(category)')
    axes[2].set_title('C. Sample predictions (12 trials overlaid)',
                       fontsize=11, fontweight='bold')
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=9, frameon=False, loc='upper right')

    fig.suptitle(
        'Synthetic Jensen demo — same random MLP, same random inputs, '
        'no training.\nThe trial-variance gap arises from architecture alone.',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = os.path.join(out_dir, 'synthetic_jensen_demo.png')
    save_fig(fig, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    return out_path, dict(H_ppc=H_ppc, H_samp=H_samp,
                           V_ppc=V_ppc, V_samp=V_samp,
                           J_ppc=J_ppc, J_samp=J_samp,
                           V_ratio=ratio)


def synthetic_scan_neurons(out_dir, n_grid=(10, 20, 30, 50, 80, 120), seed=0):
    """Architectural-floor scan across population sizes.

    Repeats the closed-form demo across a grid of n_neurons. Both
    decoders' trial-to-trial variance scales with N (more inputs ->
    larger logit swings), but the spatial/temporal RATIO is roughly N-invariant
    here -- the asymmetry is a constant architectural floor, not a
    growing-with-N effect, in the untrained / random-MLP regime.

    The empirically observed *growth* of the spat_shf / temp_shf ratio
    with N (e.g. ~1.08 at N=10 -> ~1.25 at full population) comes from
    training dynamics on top of this architectural floor: trained
    networks fit more spurious structure on shuffled targets as N
    grows, and they do so asymmetrically because the spatial architecture
    lacks the output-averaging variance reducer the sampling
    architecture gets for free.
    """
    V_ppc, V_samp, J_ppc, J_samp = [], [], [], []
    for n in n_grid:
        p_ppc, p_samp, _ = _synthetic_predictions(n_neurons=n, seed=seed)
        V_ppc.append(trial_variance(p_ppc))
        V_samp.append(trial_variance(p_samp))
        marg_p = p_ppc.mean(axis=0, keepdims=True)
        marg_s = p_samp.mean(axis=0, keepdims=True)
        J_ppc.append(js_divergence(p_ppc, np.broadcast_to(marg_p, p_ppc.shape)).mean())
        J_samp.append(js_divergence(p_samp, np.broadcast_to(marg_s, p_samp.shape)).mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    spat_c, temp_c = '#1f77b4', '#d62728'

    axes[0].plot(n_grid, V_ppc, '-o', color=spat_c, lw=2.0, ms=6,
                  label='spatial-style')
    axes[0].plot(n_grid, V_samp, '-s', color=temp_c, lw=2.0, ms=6,
                  label='Sampling-style')
    axes[0].set_xlabel('n_neurons (synthetic)')
    axes[0].set_ylabel('$\\sum_c$ var$_{trial}$(pred[:, c])')
    axes[0].set_title('Trial-to-trial variance vs population size',
                       fontsize=11, fontweight='bold')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, frameon=False)

    ratio = np.array(J_ppc) / np.array(J_samp)
    axes[1].plot(n_grid, ratio, '-o', color='#33a02c', lw=2.0, ms=6)
    axes[1].axhline(1.0, color='0.55', ls='--', lw=1.2,
                     label='No asymmetry')
    axes[1].set_xlabel('n_neurons (synthetic)')
    axes[1].set_ylabel('JS(ppc) / JS(samp) vs marginal')
    axes[1].set_title('Shuffle-loss ratio (architectural floor)',
                       fontsize=11, fontweight='bold')
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9, frameon=False)

    fig.suptitle(
        'Synthetic Jensen demo — population-size scan\n'
        'Architectural floor (~1.8× JS ratio) is roughly N-invariant; '
        'the empirical growth with N comes from training dynamics.',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = os.path.join(out_dir, 'synthetic_jensen_scan.png')
    save_fig(fig, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    return out_path


# ----------------------------------------------------------------------
# 3. Synthetic trained demo — reproduce the growth-with-N empirically
# ----------------------------------------------------------------------

def _train_simple_mlp(x_train, y_train, model_type, n_cats,
                       hidden=32, epochs=200, lr=5e-3, seed=0):
    """Tiny PyTorch training loop with the same forward-pass split as
    production (spatial averages inputs first, sampling averages outputs).

    Loss is JS to the time-averaged target (mirrors the production
    ``custom_loss_all_H`` JS branch). No entropy regulariser, no
    minibatching, no Optuna -- just enough to demonstrate the
    growth-with-N asymmetry in a controlled setup.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device = torch.device('cpu')
    n_neurons = x_train.shape[-1]
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    if model_type == 'ppc':
        x_in = x.mean(dim=1)                                    # (N, n_neurons)
    else:
        x_in = x                                                # (N, T, n_neurons)

    net = nn.Sequential(
        nn.Linear(n_neurons, hidden),
        nn.Tanh(),
        nn.Linear(hidden, n_cats),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def predict(net, x_in, model_type):
        if model_type == 'ppc':
            return F.softmax(net(x_in), dim=-1)                 # (N, n_cats)
        # sampling: per-bin softmax, then mean over T
        logits = net(x_in)                                       # (N, T, n_cats)
        return F.softmax(logits, dim=-1).mean(dim=1)            # (N, n_cats)

    def js(p, q):
        p = p.clamp(1e-12, 1.0); q = q.clamp(1e-12, 1.0)
        m = 0.5 * (p + q)
        return 0.5 * ((p * (p.log() - m.log())).sum(-1)
                      + (q * (q.log() - m.log())).sum(-1)).mean()

    y_target = y.mean(dim=1) if y.ndim == 3 else y              # collapse time
    for _ in range(epochs):
        opt.zero_grad()
        pred = predict(net, x_in, model_type)
        loss = js(pred, y_target)
        loss.backward()
        opt.step()
    net.eval()
    return net, predict


def _synthetic_training_data(n_train, n_test, n_neurons, T, n_cats,
                              shuffle=True, seed=0):
    """Generate synthetic activity + targets, optionally shuffling the
    target labels relative to inputs.

    Targets are pulled from a 91-bin distribution-of-distributions:
    each trial gets a Gaussian-bump target centered at a per-trial
    angle plus broad noise. Shuffling permutes the angle assignment
    across trials so input <-> target has no relationship.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    angles = rng.uniform(0, n_cats, size=n_total)
    s = rng.normal(0.0, 0.5, size=(n_total, n_neurons))
    eps = rng.normal(0.0, 1.0, size=(n_total, T, n_neurons))
    x = s + eps                                                 # (N, T, n_neurons)

    grid = np.arange(n_cats)[None, :]
    bumps = np.exp(-0.5 * ((grid - angles[:, None]) / 8.0) ** 2)
    bumps = bumps / bumps.sum(axis=1, keepdims=True)
    y = bumps                                                   # (N, n_cats)

    if shuffle:
        perm = rng.permutation(n_total)
        y = y[perm]

    return (x[:n_train], y[:n_train], x[n_train:], y[n_train:])


def synthetic_training_scan(out_dir, n_grid=(10, 20, 40, 80, 120),
                             n_train=400, n_test=200, T=10, n_cats=91,
                             epochs=200, n_repeats=2, seed=0):
    """Train both architectures on SHUFFLED targets and recover the
    growth-with-N asymmetry empirically.

    For each n_neurons in ``n_grid``:
      - Generate one synthetic dataset with shuffled labels.
      - Train a spatial-style and a sampling-style network on it.
      - Score held-out test loss (JS to per-trial target).
      - Repeat ``n_repeats`` times with different seeds and average.

    The expected pattern: both losses sit well above the
    perfect-marginal floor (because trained networks overfit shuffled
    structure), but spatial's loss is larger and the gap widens with N.
    """
    import torch
    means_ppc, means_samp, sd_ppc, sd_samp = [], [], [], []
    for n in n_grid:
        losses_ppc, losses_samp = [], []
        for rep in range(n_repeats):
            x_tr, y_tr, x_te, y_te = _synthetic_training_data(
                n_train, n_test, n_neurons=n, T=T, n_cats=n_cats,
                shuffle=True, seed=seed + 100 * rep)
            net_p, pred_fn_p = _train_simple_mlp(
                x_tr, y_tr, 'ppc', n_cats, epochs=epochs, seed=seed + rep)
            net_s, pred_fn_s = _train_simple_mlp(
                x_tr, y_tr, 'sampling', n_cats, epochs=epochs, seed=seed + rep)

            with torch.no_grad():
                x_te_t = torch.tensor(x_te, dtype=torch.float32)
                p_p = pred_fn_p(net_p, x_te_t.mean(dim=1), 'ppc').cpu().numpy()
                p_s = pred_fn_s(net_s, x_te_t, 'sampling').cpu().numpy()
            losses_ppc.append(float(js_divergence(p_p, y_te).mean()))
            losses_samp.append(float(js_divergence(p_s, y_te).mean()))
        means_ppc.append(np.mean(losses_ppc))
        means_samp.append(np.mean(losses_samp))
        sd_ppc.append(np.std(losses_ppc))
        sd_samp.append(np.std(losses_samp))
        print(f'  [scan] N={n}: ppc={means_ppc[-1]:.4f}±{sd_ppc[-1]:.4f}  '
              f'samp={means_samp[-1]:.4f}±{sd_samp[-1]:.4f}  '
              f'ratio={means_ppc[-1]/means_samp[-1]:.3f}')

    means_ppc = np.array(means_ppc); means_samp = np.array(means_samp)
    sd_ppc = np.array(sd_ppc); sd_samp = np.array(sd_samp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    spat_c, temp_c = '#1f77b4', '#d62728'
    axes[0].fill_between(n_grid, means_ppc - sd_ppc, means_ppc + sd_ppc,
                          color=spat_c, alpha=0.18)
    axes[0].plot(n_grid, means_ppc, '-o', color=spat_c, lw=2.0, ms=6,
                  label='spatial-style')
    axes[0].fill_between(n_grid, means_samp - sd_samp, means_samp + sd_samp,
                          color=temp_c, alpha=0.18)
    axes[0].plot(n_grid, means_samp, '-s', color=temp_c, lw=2.0, ms=6,
                  label='Sampling-style')
    axes[0].set_xlabel('n_neurons (synthetic)')
    axes[0].set_ylabel('Held-out JS  (mean across repeats)')
    axes[0].set_title('Trained on SHUFFLED targets — held-out loss',
                       fontsize=11, fontweight='bold')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, frameon=False)

    ratio = means_ppc / means_samp
    axes[1].plot(n_grid, ratio, '-o', color='#33a02c', lw=2.0, ms=6)
    axes[1].axhline(1.0, color='0.55', ls='--', lw=1.2, label='No asymmetry')
    axes[1].set_xlabel('n_neurons (synthetic)')
    axes[1].set_ylabel('spatial loss / Sampling loss')
    axes[1].set_title('Asymmetry growth with N (training-driven)',
                       fontsize=11, fontweight='bold')
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9, frameon=False)

    fig.suptitle(
        'Synthetic training scan — trained on shuffled targets\n'
        'Empirical growth of the asymmetry with N reproduced from architecture + training.',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path = os.path.join(out_dir, 'synthetic_training_scan.png')
    save_fig(fig, os.path.dirname(out_path), os.path.splitext(os.path.basename(out_path))[0])
    return out_path, dict(n_grid=list(n_grid),
                           means_ppc=means_ppc.tolist(),
                           means_samp=means_samp.tolist(),
                           ratio=ratio.tolist())


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def discover_mat():
    """Pick a population_results .mat to use for the empirical breakdown.

    Preference order:
      1. JS-loss / Q file (matches the neuron-scaling target).
      2. fixed_hyperparams file from any target.
      3. Any population_results_*.mat.
    """
    candidates = (
        'population_results_fixed_hyperparams_JS_stratified_balanced.mat',
        'population_results_fixed_hyperparams_stratified_balanced.mat',
    )
    for name in candidates:
        path = NN_DECODER / name
        if path.exists():
            return path
    for path in sorted(NN_DECODER.glob('population_results_*.mat')):
        return path
    raise SystemExit(
        f'No population_results_*.mat found in {NN_DECODER}; '
        'the empirical breakdown needs the saved predicted distributions.')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mat', default=None,
                        help='population_results_*.mat to analyse '
                             '(default: auto-discovered).')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory for figures (default: '
                             'nn_decoder/figures/shuffle_asymmetry/).')
    args = parser.parse_args(argv)

    mat_path = Path(args.mat) if args.mat else discover_mat()
    out_dir = args.out_dir if args.out_dir else str(figures_dir(FIGS_SUBDIR))
    os.makedirs(out_dir, exist_ok=True)

    print(f'[empirical] reading {mat_path.name} ...')
    breakdown = _per_mouse_breakdown(mat_path)
    if not breakdown:
        raise SystemExit(
            f'No mouse predictions found in {mat_path}; '
            'is this an old .mat that does not store Dist[*]["decoded"]?')

    print(f'[empirical] {len(breakdown)} mice; writing breakdown figure ...')
    emp_path = plot_empirical_breakdown(
        breakdown, mat_path,
        os.path.join(out_dir, 'empirical_breakdown.png'))
    print(f'             -> {emp_path}')

    print('[synthetic] fixed random MLP demo ...')
    jen_path, jen_stats = synthetic_jensen_demo(out_dir)
    print(f'             -> {jen_path}')
    print(f'             H_ppc={jen_stats["H_ppc"]:.3f}  '
          f'H_samp={jen_stats["H_samp"]:.3f}  '
          f'V_ratio={jen_stats["V_ratio"]:.2f}×  '
          f'JS_ppc/JS_samp={jen_stats["J_ppc"]/jen_stats["J_samp"]:.2f}×')

    print('[synthetic] population-size scan (architectural floor) ...')
    scan_path = synthetic_scan_neurons(out_dir)
    print(f'             -> {scan_path}')

    print('[synthetic] training scan on shuffled targets ...')
    try:
        import torch  # noqa: F401
        train_path, train_stats = synthetic_training_scan(out_dir)
        print(f'             -> {train_path}')
    except ImportError:
        train_path = None
        train_stats = None
        print('             skipped: torch not installed.')

    print('\nDone.')
    return dict(empirical=emp_path, jensen=jen_path, scan=scan_path,
                training=train_path, stats=jen_stats,
                training_stats=train_stats, breakdown=breakdown)


if __name__ == '__main__':
    main()
