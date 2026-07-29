# -*- coding: utf-8 -*-
"""How uncertain is the ideal observer about its OWN posterior? (the "noise variance")

THE QUANTITY. The target we train against, ``TrialTbl_Struct['post_s_marginal']``, is a MARGINAL:
the IO's measurement m is latent, so the exported posterior is

    pbar_t(s) = sum_m  w_t(m) * p(s | m, kappa_t)

with ``w_t(m) = IO.animals[i].inferred.m_posteriors[t]`` (181-long, sums to 1) and
``p(s|m,kappa)`` the von Mises-times-prior posterior rebuilt here from the fitted per-animal
parameters. VERIFIED IN THIS SCRIPT: that sum reproduces the exported target to ~1e-12, so the
decomposition is exact, not an approximation.

The noise variance is the spread of the summands around that mean — how much the target posterior
would MOVE had the observer drawn a different measurement:

    V_t(s) = sum_m w_t(m) * ( p(s|m,kappa_t) - pbar_t(s) )^2          # (n_trials, 91)

read two ways, as asked:
  * ACROSS ORIENTATIONS, WITHIN A TRIAL -> the profile V_t(s) over the 91 orientation bins: which
    orientations the IO is least sure about. Reported raw (SD) and relative (SD / pbar).
  * ACROSS TRIALS -> the per-trial scalar sqrt(sum_s V_t(s)) and the noise FRACTION
    sum_s V_t(s) / sum_s E_m[p^2], i.e. how much of the target's second moment is measurement
    noise rather than the mean shape. Aggregated per mouse, then mean +- SEM over the 6.

WHY IT MATTERS (Mate's proposal). The projection loss weights PC k by ``explained_var[k]`` — the
ACROSS-TRIAL variance of the targets along PC k, i.e. signal. The same decomposition gives the
WITHIN-trial, across-measurement variance along PC k,

    V_t[k] = sum_m w_t(m) * ( <p(.|m) - pbar_t, u_k> )^2

which is the noise on that axis. Panel (f) puts the two side by side; their ratio is the
signal-to-noise weight that would replace ``explained_var``. This script only MEASURES the
quantity — it does not change any loss.

CAVEAT, checked and unresolved: w_t(m) is trial-specific beyond (orientation, contrast,
dispersion) — two trials in the same stimulus cell have different w. The IO was fit in
``conf_only`` mode with a velocity-based confidence readout, so w is a genuine POSTERIOR over m
given the trial's behavioural readout, not the generative p(m|s). Attempts to reproduce it from
the exported params via a Gaussian velocity link on g(m), |g(m)| or max_s p(s|m) all failed to
match (best 2.3e-2, no better than the generative baseline). This does not affect anything here:
whatever w is, it is the weighting that defines the fitted target, and the decomposition above is
exact against it.

Outputs (PNG+SVG) under figures/io_noise/: io_noise_variance
Usage:  python diagnostics/io_noise_variance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent / 'ideal_observer' / 'io_hmm'))
import peakiness_style as ps  # noqa: E402
from io_core import (IOGrids, kappa_for_trial, prior_bimodal,  # noqa: E402
                     posterior_s_given_m)

EXPORT = Path(__file__).resolve().parents[2] / 'data' / 'VR_Decoder_Data_Export.mat'
RUN = 'hpsweep_v2'
CELL = 'lam0p003_drop0_acttanh_h8_pat0_vf0p2_wd0p0001_shp30'
C_MEAN, C_NOISE = '0.35', '#d95f02'
N_PC = 20
SUPPORT = 0.10          # relative-noise panel is masked to bins >= 10% of the trial's peak
NOISE_FLOOR = 1e-6      # PCs below this share of the noise have no usable SNR (see main)


def per_animal(animal, U=None):
    """Exact m-decomposition for one animal. Returns a dict of measured quantities."""
    g = IOGrids.default()
    prior = prior_bimodal(g, 3.0)
    d, fp, inf = animal['data'], animal['fit']['full_params'], animal['inferred']
    c = np.asarray(d['contrast'], float)
    disp = np.asarray(d['dispersion'], float)
    ori = np.asarray(d['orientation'], float)
    kap = kappa_for_trial(None, c, disp, float(fp['kappa_amp']), float(fp['c_power']),
                          float(fp['d_power']), float(fp['kappa_min']))
    W = np.asarray(inf['m_posteriors'], float)                    # (n, 181)  weights over m
    ref = np.asarray(inf['post_s_marginal'], float)               # (n, 91)   the training target

    n = W.shape[0]
    pbar = np.zeros((n, 91)); Ep2 = np.zeros((n, 91))
    Vk = np.zeros((n, U.shape[0])) if U is not None else None
    uk, inv = np.unique(np.round(kap, 12), return_inverse=True)
    for j, k in enumerate(uk):
        sel = inv == j
        P = posterior_s_given_m(g, float(k), prior)               # (181, 91)  p(s | m, kappa)
        pbar[sel] = W[sel] @ P
        Ep2[sel] = W[sel] @ (P ** 2)
        if U is not None:
            A = P @ U.T                                           # (181, n_pc)  <p(.|m), u_k>
            Vk[sel] = W[sel] @ (A ** 2) - (W[sel] @ A) ** 2
    V = np.clip(Ep2 - pbar ** 2, 0.0, None)                       # (n, 91)  the noise variance

    return dict(V=V, pbar=pbar, Ep2=Ep2, Vk=Vk, ori=ori, contrast=c, dispersion=disp, kappa=kap,
                recon_err=float(np.abs(pbar - ref).max()),
                target=ref)


def _ms(v, axis=0):
    """nan-safe mean and SEM (the relative-noise profile is masked off its support)."""
    v = np.asarray(v, float)
    n = np.maximum(np.sum(~np.isnan(v), axis=axis), 1)
    with np.errstate(invalid='ignore'):
        return np.nanmean(v, axis), np.nanstd(v, axis=axis, ddof=1) / np.sqrt(n)


def main(results_root, out_root):
    ps.apply()
    M = sio.loadmat(str(EXPORT), simplify_cells=True)
    animals = M['IO']['animals']

    # PC basis + across-trial (signal) weights, straight from the trained cell — loss-invariant.
    res = sio.loadmat(str(Path(results_root) / RUN / CELL / 'Q_PCA_half_100ms_all' /
                          'stratified_balanced.mat'), simplify_cells=True)['results']

    per, evar = [], []
    print('EXACTNESS OF THE DECOMPOSITION  target == sum_m w(m) p(s|m)')
    for i, a in enumerate(animals):
        Dm = res[f'mouse_{i}']['Dist']
        U = np.asarray(Dm['pcs'], float)                          # (91, 91) rows = PCs
        ev = np.asarray(Dm['explained_var'], float)
        # the stored evar carries shape_lambda's floor; undo it to recover raw signal variance
        ev = np.clip(ev - 30.0 / 100.0, 0.0, None)
        ev = ev / ev.sum()
        r = per_animal(a, U)
        # trial alignment check: every decoder target must exist among this animal's IO targets
        dt = np.asarray(Dm['spat']['target'], float)
        hit = np.abs(r['target'][None, :, :] - dt[:, None, :]).sum(-1).min(1)
        print(f'  animal {i}  n={r["V"].shape[0]:4d}  max|recon-target| = {r["recon_err"]:.2e}'
              f'   decoder-target match (max L1 to nearest IO row) = {hit.max():.2e}')
        per.append(r); evar.append(ev)

    # ---- the two numbers asked for -----------------------------------------------------------
    sd_prof = np.stack([np.sqrt(p['V']).mean(0) for p in per])                    # (6, 91)
    pbar_prof = np.stack([p['pbar'].mean(0) for p in per])
    # Relative noise is only meaningful where the target carries mass: in the far tails both
    # pbar and V are ~0 and their ratio is numerically unbounded (it reached 1.3e4 unmasked,
    # driven entirely by bins with < 1e-6 probability). Mask to the support: bins at or above
    # 10% of that trial's peak.
    with np.errstate(invalid='ignore'):
        cv_prof = np.stack([
            np.nanmean(np.where(p['pbar'] >= SUPPORT * p['pbar'].max(1, keepdims=True),
                                np.sqrt(p['V']) / np.clip(p['pbar'], 1e-12, None), np.nan), 0)
            for p in per])                                                        # (6, 91)
    tot_sd = [np.sqrt(p['V'].sum(1)) for p in per]                               # per trial
    frac = [p['V'].sum(1) / np.clip(p['Ep2'].sum(1), 1e-12, None) for p in per]   # noise fraction

    print('\nWITHIN A TRIAL, ACROSS THE 91 ORIENTATIONS  (mean over trials, then over 6 mice)')
    m_, s_ = _ms(sd_prof)
    print(f'  noise SD averaged over orientations : {m_.mean():.5f}')
    print(f'  peak of the SD profile              : {m_.max():.5f} at {int(np.argmax(m_))} deg')
    print(f'  trough                              : {m_.min():.5f} at {int(np.argmin(m_))} deg')
    mc = np.nanmean(cv_prof, 0)
    print(f'  relative (SD / target), on the support: {np.nanmean(mc):.2f}x  '
          f'(range {np.nanmin(mc):.2f}-{np.nanmax(mc):.2f} over orientation)')
    l2 = np.array([np.mean(np.sqrt(p['V'].sum(1)) / np.sqrt((p['pbar'] ** 2).sum(1)))
                   for p in per])
    print(f'  mass-weighted  ||noise|| / ||target||: {l2.mean():.3f} +- '
          f'{l2.std(ddof=1)/np.sqrt(6):.3f}   (the scale-free headline number)')

    print('\nACROSS TRIALS  (per-mouse mean, then mean +- SEM over 6)')
    tm, ts = _ms(np.array([t.mean() for t in tot_sd]))
    fm, fs = _ms(np.array([f.mean() for f in frac]))
    print(f'  total noise SD  sqrt(sum_s V)       : {tm:.4f} +- {ts:.4f}')
    print(f'  noise FRACTION  sum_s V / sum_s E[p^2]: {fm:.3f} +- {fs:.3f}   '
          f'(1 = the target is pure measurement noise, 0 = every m gives the same posterior)')
    cvs = np.array([t.std(ddof=1) / t.mean() for t in tot_sd])
    print(f'  spread across trials (CV of the scalar): {cvs.mean():.2f} +- '
          f'{cvs.std(ddof=1)/np.sqrt(6):.2f}')

    # ---- PC decomposition ---------------------------------------------------------------------
    Vk = np.stack([p['Vk'].mean(0) for p in per])                                # (6, 91) noise
    Vk_n = Vk / Vk.sum(1, keepdims=True)
    ev = np.stack(evar)                                                          # (6, 91) signal
    chk = np.array([abs(p['Vk'].sum(1).mean() - p['V'].sum(1).mean()) for p in per]).max()
    print(f'\nPC DECOMPOSITION  (sum_k V[k] == sum_s V(s) check: max abs diff {chk:.2e})')
    print(f"{'PC':>4s}{'signal evar':>13s}{'noise frac':>12s}{'signal/noise':>14s}")
    snr = ev.mean(0) / np.clip(Vk_n.mean(0), 1e-15, None)
    for k in range(8):
        print(f'{k+1:4d}{ev.mean(0)[k]:13.4f}{Vk_n.mean(0)[k]:12.4f}{snr[k]:14.2f}')
    print(f'  cumulative over first {N_PC} PCs: signal {ev.mean(0)[:N_PC].sum():.3f}, '
          f'noise {Vk_n.mean(0)[:N_PC].sum():.3f}')

    # What would noise-normalisation DO to the loss weights, versus shape_lambda?
    # CONDITIONING. The ratio evar/noise is unusable raw: past ~PC30 both terms are float dust,
    # and dividing dust by dust hands the entire weight vector to the numerically empty tail
    # (PC1's share collapses to 0.000). The ratio is only defined where the noise estimate is
    # real, so it is computed on the PCs carrying >= NOISE_FLOOR of the noise and zero elsewhere.
    w_evar = ev.mean(0)
    live = Vk_n.mean(0) >= NOISE_FLOOR
    w_snr = np.where(live, np.clip(snr, 0, None), 0.0); w_snr = w_snr / w_snr.sum()
    w_shape = w_evar + 30.0 / 100.0; w_shape = w_shape / w_shape.sum()
    print(f'\nIMPLIED PROJECTION-LOSS WEIGHTS  ({int(live.sum())}/91 PCs have a usable noise '
          f'estimate at floor {NOISE_FLOOR:g})')
    print(f"{'PC':>4s}{'evar (current)':>16s}{'evar/noise':>13s}{'shape_lambda=30':>18s}")
    for k in range(6):
        print(f'{k+1:4d}{w_evar[k]:16.4f}{w_snr[k]:13.4f}{w_shape[k]:18.4f}')
    for nm, w in [('evar', w_evar), ('evar/noise', w_snr), ('shape30', w_shape)]:
        print(f'  {nm:12s} PC1 share {w[0]:.3f}   effective #PCs (1/sum w^2) '
              f'{1.0 / (w ** 2).sum():6.2f}')

    # ---- figure -------------------------------------------------------------------------------
    x = np.arange(91)
    fig, ax = plt.subplots(2, 3, figsize=ps.figsize(3, 2), constrained_layout=True)

    # (a) what the noise IS: one broad and one sharp trial from mouse 0
    p0 = per[0]
    g = IOGrids.default(); prior = prior_bimodal(g, 3.0)
    tot0 = np.sqrt(p0['V'].sum(1))
    for t, ls, nm in [(int(np.argmax(tot0)), '-', 'most uncertain trial'),
                      (int(np.argmin(tot0)), '--', 'least uncertain trial')]:
        sd = np.sqrt(p0['V'][t])
        ax[0][0].plot(x, p0['pbar'][t], color=C_MEAN, ls=ls, lw=1.8)
        ax[0][0].fill_between(x, np.clip(p0['pbar'][t] - sd, 0, None), p0['pbar'][t] + sd,
                              color=C_NOISE, alpha=0.28 if ls == '-' else 0.14, lw=0)
    ax[0][0].set_ylabel('probability', fontsize=8)
    ax[0][0].set_title('target ± 1 SD over measurements', fontsize=9)
    ax[0][0].legend(handles=[
        Line2D([0], [0], color=C_MEAN, lw=1.8, label='target, most uncertain trial'),
        Line2D([0], [0], color=C_MEAN, lw=1.8, ls='--', label='target, least uncertain trial'),
        plt.Rectangle((0, 0), 1, 1, fc=C_NOISE, alpha=0.28, ec='none', label='± 1 SD over m')],
        fontsize=6, frameon=True)

    # (b) profile across orientations, raw
    def band(a_, prof, col, lab):
        m_, s_ = _ms(prof)
        a_.plot(x, m_, color=col, lw=1.7, label=lab)
        a_.fill_between(x, m_ - s_, m_ + s_, color=col, alpha=0.25, lw=0)

    band(ax[0][1], pbar_prof, C_MEAN, 'mean target')
    band(ax[0][1], sd_prof, C_NOISE, 'noise SD')
    ax[0][1].set_ylabel('probability', fontsize=8)
    ax[0][1].set_title('within trial, across orientations', fontsize=9)

    # (c) relative, on the support only
    band(ax[0][2], cv_prof, C_NOISE, f'noise SD ÷ target (≥{SUPPORT:.0%} of peak)')
    ax[0][2].axhline(1.0, color='0.4', ls=':', lw=1.2)
    ax[0][2].set_ylabel('relative noise SD', fontsize=8)
    ax[0][2].set_title('relative to the target', fontsize=9)

    for a_ in ax[0]:
        a_.set_xlabel('orientation (deg)', fontsize=8)
    ax[0][1].legend(fontsize=6, frameon=True)
    ax[0][2].legend(fontsize=6, frameon=True)

    # (d) across trials, vs stimulus orientation, split by contrast
    ucon = np.unique(np.concatenate([p['contrast'] for p in per]))
    cols = plt.cm.viridis(np.linspace(0.1, 0.88, ucon.size))
    uori = np.unique(np.concatenate([p['ori'] for p in per]))
    for cc, col in zip(ucon, cols):
        Y = np.full((len(per), uori.size), np.nan)
        for i, p in enumerate(per):
            for j, o in enumerate(uori):
                sel = (p['contrast'] == cc) & (p['ori'] == o)
                if sel.sum():
                    Y[i, j] = np.sqrt(p['V'].sum(1))[sel].mean()
        keep = np.sum(~np.isnan(Y), 0) > 1
        m_, s_ = _ms(Y)
        ax[1][0].errorbar(uori[keep], m_[keep], yerr=s_[keep],
                          color=col, lw=1.5, marker='o', ms=3.5, capsize=2, label=f'c={cc:g}')
    ax[1][0].set_xlabel('stimulus orientation (deg)', fontsize=8)
    ax[1][0].set_ylabel('total noise SD', fontsize=8)
    ax[1][0].set_title('across trials, by stimulus', fontsize=9)

    # (e) noise fraction vs sensory precision. Contrast and dispersion are NOT crossed in this
    # design (each contrast appears at only two dispersions, and they differ across contrasts),
    # so plotting against either alone is confounded; kappa is the single quantity both feed.
    udis = np.unique(np.concatenate([p['dispersion'] for p in per]))
    mks = dict(zip(udis, ['o', 's', '^', 'D', 'v']))
    for i, p in enumerate(per):
        for cc, col in zip(ucon, cols):
            for dd in udis:
                sel = (p['contrast'] == cc) & (p['dispersion'] == dd)
                if sel.sum() < 5:
                    continue
                ax[1][1].plot(p['kappa'][sel][0], frac[i][sel].mean(), mks[dd], color=col,
                              ms=4.5, mec='k', mew=0.4, alpha=0.85)
    ax[1][1].set_xscale('log')
    ax[1][1].set_xlabel('sensory precision κ', fontsize=8)
    ax[1][1].set_ylabel('noise fraction of second moment', fontsize=8)
    ax[1][1].set_title('one point per mouse × condition', fontsize=9)
    ax[1][1].legend(handles=(
        [Line2D([0], [0], ls='', marker='o', color=c_, ms=4.5, mec='k', mew=0.4,
                label=f'c={cc:g}') for cc, c_ in zip(ucon, cols)] +
        [Line2D([0], [0], ls='', marker=mks[dd], color='0.6', ms=4.5, mec='k', mew=0.4,
                label=f'disp={dd:g}') for dd in udis]), fontsize=5.5, frameon=True, ncol=2)

    # (f) PC decomposition: signal (evar) vs noise, same basis
    kk = np.arange(1, N_PC + 1)
    m_, s_ = _ms(ev[:, :N_PC])
    ax[1][2].errorbar(kk, m_, yerr=s_, color='#1b6ca8', lw=1.6, marker='o', ms=3.5, capsize=2,
                      label='signal (across-trial evar)')
    m_, s_ = _ms(Vk_n[:, :N_PC])
    ax[1][2].errorbar(kk, m_, yerr=s_, color=C_NOISE, lw=1.6, marker='s', ms=3.5, capsize=2,
                      label='noise (across-m, normalised)')
    ax[1][2].set_yscale('log')
    ax[1][2].set_xlabel('principal component', fontsize=8)
    ax[1][2].set_ylabel('variance fraction', fontsize=8)
    ax[1][2].set_title('signal vs noise per PC', fontsize=9)

    ax[1][0].legend(fontsize=6, frameon=True)
    ax[1][2].legend(fontsize=6, frameon=True)
    ps.label_panels(ax.ravel())
    ps.save_fig(fig, Path(out_root), 'io_noise_variance')
    print(f'\nDone -> {Path(out_root).resolve()}/io_noise_variance.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/io_noise')
    a = ap.parse_args()
    main(a.results_root, a.out_root)
