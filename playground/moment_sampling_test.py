# -*- coding: utf-8 -*-
"""Quick test: CAN the temporal (sampling) decoder produce individual bin
posteriors that are SHARP and scattered so their moments reproduce Q's mean and
variance — on real data?  (2026-08-04, Theo's request.)

Law of total variance for per-bin posteriors p_t (mean mu_t, variance s2_t):

    sigma2_Q  =  Var_t[mu_t]        +      E_t[s2_t]
                 (across-bin scatter)      (within-bin width)

PPC puts all of it within-bin (broad identical bins, no scatter). Genuine
SAMPLING puts it across-bin (sharp bins scattered by sigma_Q). Both reproduce Q's
variance on the time-average — they differ only in the PARTITION. This script
trains the sampling decoder with a moment-matching objective that explicitly
pushes for the SAMPLING partition, and asks whether the real data supports it.

Objective (per trial, on the per-bin means mu_t and variances s2_t):
    L_mean    = ((mean_t mu_t - mu_Q) / R)^2                 # centre on Q's mean
    L_scatter = ((Var_t[mu_t] - sigma2_Q) / sigma2_Q)^2      # scatter = full variance
    L_sharp   = mean_t s2_t / sigma2_Q                       # within-bin -> 0 (sharp)
    L = L_mean + L_scatter + lam_sharp * L_sharp
(R = 90 deg range; mu_Q, sigma2_Q are the linear moments of the trial's Q.)

A standard Jensen-KL head is trained on the SAME split as the reference (the PPC
solution the diagnostic already showed: ~Q-wide bins, no scatter).

THE decisive readout is not the marginal moments (the loss forces those if the
model has any freedom) but whether per-trial SCATTER TRACKS per-trial sigma_Q on
HELD-OUT trials. Fixed scatter that matches the marginal = moment-faking; scatter
that rises with the trial's posterior width = genuine trial-conditioned sampling.

Leakage-safe: same stratified split, z-scoring, and target as production
(replicated from run_experiment). Standalone — touches no load-bearing code.
CPU, ~seconds/mouse.

Usage:  OMP_NUM_THREADS=1 python playground/moment_sampling_test.py
        OMP_NUM_THREADS=1 python playground/moment_sampling_test.py --mice 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NN = str(Path(__file__).resolve().parent.parent / 'nn_decoder')
sys.path.insert(0, NN)
from utils import (load_vr_export, apply_temporal_binning,          # noqa: E402
                   get_stratified_train_test_indices)
from nn_classifier import SimpleFlexibleNNClassifier                 # noqa: E402
from training.targets import make_target                            # noqa: E402

GRID = np.linspace(0.0, 90.0, 91)          # orientation grid (folded, degrees)
R = 90.0                                    # range, for normalising the mean term


# --------------------------------------------------------------- data (leakage-safe)
def load_mouse(mouse_id, window='half', bin_ms=100):
    acts, tperc, tdec, tlik, trials = load_vr_export(mouse_id)     # acts (N, trials, xG)
    act_t = np.transpose(acts, (1, 2, 0))                          # (trials, xG, N)
    binned = apply_temporal_binning(act_t, time_window=window, bin_size_ms=bin_ms)
    binned = np.transpose(binned, (0, 2, 1))                       # (trials, T, N)
    Q = np.asarray(make_target('perception', trials, targets_perc=tperc), float)
    Q = Q / Q.sum(1, keepdims=True)
    conds = np.array(list(zip(trials['orientation'], trials['contrast'],
                              trials['dispersion'])))
    _, cats = np.unique(conds, axis=0, return_inverse=True)
    tr, te = get_stratified_train_test_indices(cats, test_size=0.5, random_state=42)
    mu = binned[tr].mean((0, 1), keepdims=True)
    sd = binned[tr].std((0, 1), keepdims=True)
    sd[sd == 0] = 1.0
    binned = (binned - mu) / sd
    return dict(X=binned, Q=Q, tr=tr, te=te)


def moments(p):
    """Linear mean and variance over the orientation grid. p: (..., 91) torch."""
    g = torch.tensor(GRID, dtype=p.dtype, device=p.device)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    mu = (p * g).sum(-1)
    var = (p * (g - mu.unsqueeze(-1)) ** 2).sum(-1)
    return mu, var


# ------------------------------------------------------------------- training
def train(X, Q, idx, objective, hidden=(8,), epochs=500, seed=0, lam_sharp=1.0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X[idx], dtype=torch.float32)                 # (B, T, N)
    Qt = torch.tensor(Q[idx], dtype=torch.float32)                 # (B, 91)
    muQ, varQ = moments(Qt)                                        # (B,)
    model = SimpleFlexibleNNClassifier(X.shape[2], list(hidden), 91,
                                       activation='tanh')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        p = F.softmax(model(Xt), dim=-1)                           # (B, T, 91)
        if objective == 'moment':
            mu_t, s2_t = moments(p)                                # (B, T)
            mbar = mu_t.mean(1)
            scatter = mu_t.var(1, unbiased=False)
            L = (((mbar - muQ) / R) ** 2).mean() \
                + (((scatter - varQ) / varQ.clamp_min(1.0)) ** 2).mean() \
                + lam_sharp * (s2_t.mean(1) / varQ.clamp_min(1.0)).mean()
        else:  # 'jensen' — standard KL(mean_t p_t || Q)
            pbar = p.mean(1).clamp_min(1e-12)
            L = (-(Qt * torch.log(pbar)).sum(1) + (Qt * torch.log(Qt.clamp_min(1e-12))).sum(1)).mean()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, Q, idx):
    Xt = torch.tensor(X[idx], dtype=torch.float32)
    Qt = torch.tensor(Q[idx], dtype=torch.float32)
    p = F.softmax(model(Xt), dim=-1)                               # (B, T, 91)
    mu_t, s2_t = moments(p)                                        # (B, T)
    muQ, varQ = moments(Qt)                                        # (B,)
    within = s2_t.mean(1)                                          # E_t s2_t  (B,)
    scatter = mu_t.var(1, unbiased=False)                          # Var_t mu_t (B,)
    mbar = mu_t.mean(1)
    pbar = p.mean(1).clamp_min(1e-12)                              # Jensen average
    kl = (-(Qt * torch.log(pbar)).sum(1) + (Qt * torch.log(Qt.clamp_min(1e-12))).sum(1))
    # predict-mean null KL (marginal mean over these trials) for a chance ref
    pm = Qt.mean(0, keepdim=True).clamp_min(1e-12)
    kl_pm = (-(Qt * torch.log(pm)).sum(1) + (Qt * torch.log(Qt.clamp_min(1e-12))).sum(1))
    return dict(
        within_sd=np.sqrt(within.numpy()), scatter_sd=np.sqrt(scatter.numpy()),
        sigma_Q=np.sqrt(varQ.numpy()), mbar=mbar.numpy(), muQ=muQ.numpy(),
        kl=kl.numpy(), kl_norm=float(kl.mean() / kl_pm.mean()),
        avg_maxprob=float(pbar.max(1).values.mean()),
        tgt_maxprob=float(Qt.max(1).values.mean()))


def _corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan, np.nan
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    slope = np.polyfit(b[ok], a[ok], 1)[0]     # scatter_sd ~ slope * sigma_Q
    return r, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mice', nargs='+', type=int, default=list(range(6)))
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lam-sharp', type=float, default=1.0)
    ap.add_argument('--fig', default=None, help='path to save a per-mouse figure')
    a = ap.parse_args()

    print(f"Moment-matching sampling test — {len(a.mice)} mice, {a.epochs} ep, "
          f"lam_sharp={a.lam_sharp}\nGrid 0-90 deg; all stats on HELD-OUT test trials.\n")
    hdr = (f"{'mouse':>5s}{'obj':>8s}{'withinSD':>9s}{'scatterSD':>10s}{'sigmaQ':>8s}"
           f"{'avgMax':>8s}{'tgtMax':>8s}{'KLnorm':>8s}{'r(sc,sQ)':>9s}{'slope':>7s}")
    print(hdr)
    print('-' * len(hdr))
    agg = {'moment': [], 'jensen': []}
    for mid in a.mice:
        d = load_mouse(mid)
        for obj in ('moment', 'jensen'):
            m = train(d['X'], d['Q'], d['tr'], obj, epochs=a.epochs,
                      lam_sharp=a.lam_sharp)
            ev = evaluate(m, d['X'], d['Q'], d['te'])
            r, slope = _corr(ev['scatter_sd'], ev['sigma_Q'])
            agg[obj].append((ev['within_sd'].mean(), ev['scatter_sd'].mean(),
                             ev['sigma_Q'].mean(), ev['avg_maxprob'],
                             ev['tgt_maxprob'], ev['kl_norm'], r, slope))
            print(f"{mid:>5d}{obj:>8s}{ev['within_sd'].mean():9.1f}"
                  f"{ev['scatter_sd'].mean():10.1f}{ev['sigma_Q'].mean():8.1f}"
                  f"{ev['avg_maxprob']:8.4f}{ev['tgt_maxprob']:8.4f}"
                  f"{ev['kl_norm']:8.3f}{r:9.2f}{slope:7.2f}")
    print('-' * len(hdr))
    for obj in ('moment', 'jensen'):
        A = np.array(agg[obj])
        print(f"{'MEAN':>5s}{obj:>8s}{A[:,0].mean():9.1f}{A[:,1].mean():10.1f}"
              f"{A[:,2].mean():8.1f}{A[:,3].mean():8.4f}{A[:,4].mean():8.4f}"
              f"{A[:,5].mean():8.3f}{np.nanmean(A[:,6]):9.2f}{np.nanmean(A[:,7]):7.2f}")
    print("\nRead: withinSD -> 0 and scatterSD -> sigmaQ = the SAMPLING partition. "
          "r(sc,sQ) & slope on held-out trials = does per-trial scatter track the "
          "trial's posterior width (genuine sampling) or is it fixed (moment-faking)?")

    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        Am, Aj = np.array(agg['moment']), np.array(agg['jensen'])
        x = np.arange(len(a.mice))
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        # (a) bin sharpness: within-bin SD, moment vs jensen, vs Q's SD
        ax[0].plot(x, Am[:, 0], 'o', ms=6, color='#2171b5', label='moment (sharp target)')
        ax[0].plot(x, Aj[:, 0], 's', ms=6, mfc='none', color='#e6550d', label='standard KL')
        ax[0].plot(x, Am[:, 2], '_', ms=14, mew=2, color='k', label="Q's own SD")
        ax[0].set_ylabel('within-bin SD  (deg)'); ax[0].set_title('(a) can bins be made sharp?')
        ax[0].set_xlabel('mouse'); ax[0].set_xticks(x); ax[0].legend(fontsize=7)
        # (b) across-bin scatter reached vs required (sigmaQ)
        ax[1].plot(x, Am[:, 1] / Am[:, 2], 'o', ms=6, color='#2171b5', label='moment')
        ax[1].plot(x, Aj[:, 1] / Aj[:, 2], 's', ms=6, mfc='none', color='#e6550d', label='standard KL')
        ax[1].axhline(1.0, color='k', ls=':', lw=1.4)
        ax[1].set_ylabel('across-bin scatter / sigma_Q'); ax[1].set_ylim(0, 1.15)
        ax[1].set_title('(b) does scatter reach Q width?\n(1.0 = genuine sampling)')
        ax[1].set_xlabel('mouse'); ax[1].set_xticks(x); ax[1].legend(fontsize=7)
        # (c) does the time-average still reproduce Q? normalised KL
        ax[2].plot(x, Am[:, 5], 'o', ms=6, color='#2171b5', label='moment')
        ax[2].plot(x, Aj[:, 5], 's', ms=6, mfc='none', color='#e6550d', label='standard KL')
        ax[2].axhline(1.0, color='k', ls=':', lw=1.4, label='chance')
        ax[2].set_yscale('log'); ax[2].set_ylabel('KL(avg || Q) / predict-mean')
        ax[2].set_title('(c) does the time-average match Q?\n(<1 beats chance)')
        ax[2].set_xlabel('mouse'); ax[2].set_xticks(x); ax[2].legend(fontsize=7)
        fig.suptitle('Can the temporal decoder SAMPLE on real data? Bins can be made sharp (a), but the '
                     'across-bin scatter caps well below Q width (b),\nso forcing sharp bins destroys the '
                     'posterior (c). The variance lives within-instant (PPC), not across-time (SBC). n=6 mice.',
                     fontsize=9)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(f'{a.fig}.{ext}', dpi=140, bbox_inches='tight')
        print(f"  saved {a.fig}.png/.svg")


if __name__ == '__main__':
    main()
