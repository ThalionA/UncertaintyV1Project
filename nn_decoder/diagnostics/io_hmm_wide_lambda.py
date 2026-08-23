# -*- coding: utf-8 -*-
"""Entropy-lambda figures for the io_hmm_v3 width sweep — TEMPORAL decoder only.

lambda_H is the weight on ``mean H(per-bin predicted posterior)`` added to the
fit loss (``nn_classifier.custom_loss_all_H``), and it only enters the temporal
(per-bin 'sampling') decoder: the five lambda cells of every (loss, arch, mouse)
group are exact SPATIAL replicates (re-asserted here from cells.csv, max
deviation must be 0), so the spatial decoder is not plotted.

Inputs: figures/io_hmm_wide/cells.csv (written by diagnostics/io_hmm_wide_extract.py;
every metric there is the settled scorecard metric — see that file's docstring).

Figures (figures/io_hmm_wide/):
  lambda_kl_skill_grid    KL skill vs lambda_H, 5 losses x 7 archs, per-mouse
                          lines + bold median; y = mean KL(tgt||dec) / mean
                          KL(tgt||LOO predict-mean) on held-out trials (1 = null)
  lambda_s_hat_grid       the same for s_hat (equivalent sharpening; 1 = none);
                          clamped inversions (ladder end, value is a bound) are
                          drawn as hollow markers
  lambda_inertness_heatmaps   per (loss, arch): median over mice of the relative
                          range (max-min)/min of the metric across the 5 lambdas,
                          plus the number of mice whose range exceeds 5% — one
                          heatmap for KL skill, one for s_hat

Prior (e) scoring (PREDICTIONS.md 2026-08-22) printed on stdout:
  e1  lambda inert for the calibrated losses (kl/js/ce/pcaflat) at ALL widths —
      threshold: median-over-mice worst relative range of KL skill <= 5%
  e2  evar instability grows with lambda AND with width — per-mouse sign counts

Usage (from nn_decoder/):
    python diagnostics/io_hmm_wide_lambda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import peakiness_style as ps            # noqa: E402
from figsave import save_fig            # noqa: E402

OUT_DIR = HERE.parent / 'figures' / 'io_hmm_wide'
CSV = OUT_DIR / 'cells.csv'

LOSS_ORDER = ('kl', 'js', 'ce', 'pcaflat', 'pca')
LOSS_LABEL = {'kl': 'KL', 'js': 'JS', 'ce': 'CE',
              'pcaflat': 'projection (flat)', 'pca': 'projection (evar)'}
LOSS_COLOR = {'kl': ps.KL, 'js': ps.JS, 'ce': ps.CE,
              'pcaflat': ps.FLAT_EVAR, 'pca': ps.PCA_EVAR}
CALIBRATED = ('kl', 'js', 'ce', 'pcaflat')
ARCH_ORDER = ('lin', 'rr8', 'h4', 'h8', 'h16', 'h32', 'h64')
ARCH_LABEL = {'lin': 'lin  (Linear n→72)', 'rr8': 'rr8  (rank-8, no nonlin.)',
              'h4': 'h4  (tanh)', 'h8': 'h8  (tanh)', 'h16': 'h16  (tanh)',
              'h32': 'h32  (tanh)', 'h64': 'h64  (tanh)'}
WIDTH_ARCHS = ('h4', 'h8', 'h16', 'h32', 'h64')      # the width axis proper
LAMS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3)
LAM_LABEL = ('0', '1e-4', '3e-4', '1e-3', '3e-3')
MICE = tuple(range(6))
# six distinguishable mouse colours, kept away from the loss-family palette
MOUSE_COLOR = ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02')
INERT_THR = 0.05     # "inert" = worst relative range over lambda <= 5 %


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load():
    d = pd.read_csv(CSV)
    assert len(d) == 2100, len(d)
    d['lambda_H'] = d['lambda_H'].astype(float)
    assert set(np.round(d.lambda_H.unique(), 9)) == set(np.round(LAMS, 9))
    # spatial is replicates across lambda: measure it, don't assume it
    sp = d[d.decoder == 'spat']
    dev = {}
    for m in ('kl_skill', 'proj_skill', 's_hat', 'w_in_norm', 'best_epoch'):
        g = sp.groupby(['loss_family', 'arch', 'mouse'])[m]
        dev[m] = float((g.max() - g.min()).max())
    worst = max(dev.values())
    print('spatial replicate check (max over 210 groups of max-min across 5 lambdas):',
          {k: f'{v:.2e}' for k, v in dev.items()}, '->', 'PASS' if worst == 0 else 'FAIL')
    assert worst == 0.0, dev
    t = d[d.decoder == 'temp'].copy()
    assert len(t) == 1050
    return t


def grid(t, metric):
    """-> array[loss, arch, mouse, lam] of the metric (temporal decoder)."""
    a = np.full((len(LOSS_ORDER), len(ARCH_ORDER), len(MICE), len(LAMS)), np.nan)
    pv = t.pivot_table(index=['loss_family', 'arch', 'mouse'], columns='lambda_H',
                       values=metric)
    for i, lf in enumerate(LOSS_ORDER):
        for j, ar in enumerate(ARCH_ORDER):
            for k, m in enumerate(MICE):
                row = pv.loc[(lf, ar, m)]
                a[i, j, k] = [row[l] for l in sorted(row.index)]
    assert np.isfinite(a).all()
    return a


def rel_range(a):
    """(max - min) / min across the last (lambda) axis."""
    return (a.max(-1) - a.min(-1)) / a.min(-1)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def _row_scale(vals):
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    log = hi / lo > 4
    if log:
        return True, (lo / 1.15, hi * 1.15)
    pad = 0.08 * (hi - lo + 1e-9)
    return False, (min(lo - pad, 0.95), max(hi + pad, 1.05))


def fig_grid(a, stem, ylabel, suptitle, ref=1.0, clamped=None):
    nr, nc = len(LOSS_ORDER), len(ARCH_ORDER)
    fig, axes = plt.subplots(nr, nc, figsize=(14.5, 11.0), sharex=True)
    x = np.arange(len(LAMS))
    for i, lf in enumerate(LOSS_ORDER):
        log, ylim = _row_scale(a[i])
        for j, ar in enumerate(ARCH_ORDER):
            ax = axes[i, j]
            ax.axhline(ref, ls=':', lw=1.0, color=ps.CHANCE_GREY, zorder=0)
            for k in MICE:
                ax.plot(x, a[i, j, k], '-', lw=0.9, alpha=0.75, color=MOUSE_COLOR[k],
                        marker='o', ms=2.6, mec='none', zorder=2)
                if clamped is not None:
                    cl = clamped[i, j, k].astype(bool)
                    if cl.any():
                        ax.plot(x[cl], a[i, j, k][cl], 'o', ms=5.5, mfc='white',
                                mec=MOUSE_COLOR[k], mew=1.1, zorder=3)
            ax.plot(x, np.median(a[i, j], axis=0), '-', lw=2.4, color='k', zorder=4)
            if log:
                ax.set_yscale('log')
                ticks = [v for v in (0.3, 0.5, 1, 2, 3, 5, 10, 20, 30)
                         if ylim[0] <= v <= ylim[1]]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{v:g}' for v in ticks])
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(*ylim)
            ax.set_xticks(x)
            ax.set_xticklabels(LAM_LABEL, fontsize=7.5, rotation=45)
            ax.tick_params(axis='y', labelsize=7.5)
            ax.grid(True, axis='y', lw=0.4, alpha=0.35)
            if j > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(f'{LOSS_LABEL[lf]}\n{ylabel}', fontsize=8.5,
                              color=LOSS_COLOR[lf])
            if i == 0:
                ax.set_title(ARCH_LABEL[ar], fontsize=9)
            if i == nr - 1 and j == nc // 2:
                ax.set_xlabel('λ_H (weight on mean per-bin posterior entropy in the '
                              'training loss; categorical axis)', fontsize=8.5)
            ax.set_xlim(-0.4, len(LAMS) - 0.6)
        # share y across the row
        for j in range(1, nc):
            axes[i, j].sharey(axes[i, 0])
    handles = [Line2D([0], [0], color=MOUSE_COLOR[k], lw=1.2, marker='o', ms=3,
                      label=f'mouse {k}') for k in MICE]
    handles += [Line2D([0], [0], color='k', lw=2.4, label='median of 6 mice'),
                Line2D([0], [0], ls=':', color=ps.CHANCE_GREY, lw=1,
                       label=f'reference = {ref:g}')]
    if clamped is not None:
        handles.append(Line2D([0], [0], marker='o', ls='none', mfc='white', mec='0.3',
                              mew=1.1, ms=5.5,
                              label='s_hat clamped (ladder end; value is a bound)'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles),
               fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=10.5)
    save_fig(fig, OUT_DIR, stem)


def fig_heatmaps(rr_kl, rr_sh, n_kl, n_sh):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    norm = LogNorm(vmin=1e-3, vmax=3.0)
    for ax, rr, nn, name in ((axes[0], rr_kl, n_kl, 'KL skill'),
                             (axes[1], rr_sh, n_sh, 's_hat')):
        im = ax.imshow(np.clip(rr, 1e-3, None), cmap='YlOrRd', norm=norm, aspect='auto')
        for i in range(rr.shape[0]):
            for j in range(rr.shape[1]):
                v = rr[i, j]
                dark = v > 0.25
                ax.text(j, i - 0.12, f'{100 * v:.1f}%' if v < 0.1 else f'{100 * v:.0f}%',
                        ha='center', va='center', fontsize=9.5, fontweight='bold',
                        color='white' if dark else 'black')
                ax.text(j, i + 0.25, f'{nn[i, j]}/6 > {100 * INERT_THR:.0f}%',
                        ha='center', va='center', fontsize=7,
                        color='white' if dark else '0.25')
        ax.set_xticks(range(len(ARCH_ORDER)))
        ax.set_xticklabels(ARCH_ORDER)
        ax.set_yticks(range(len(LOSS_ORDER)))
        ax.set_yticklabels([LOSS_LABEL[l] for l in LOSS_ORDER])
        for lab, lf in zip(ax.get_yticklabels(), LOSS_ORDER):
            lab.set_color(LOSS_COLOR[lf])
        ax.set_xlabel('architecture (temporal decoder)')
        ax.set_title(f'{name}: relative range across the 5 λ_H\n'
                     f'(max−min)/min per mouse, median over 6 mice', fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label('median relative range over λ_H (log colour)', fontsize=8)
    cb.set_ticks([1e-3, 1e-2, INERT_THR, 1e-1, 1.0, 3.0])
    cb.set_ticklabels(['0.1%', '1%', f'{100 * INERT_THR:.0f}% (inert threshold)',
                       '10%', '100%', '300%'])
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(INERT_THR, color='k', lw=1.2)
    fig.suptitle('Is λ_H inert? Temporal decoder only (spatial cells are exact replicates '
                 'across λ_H). Small number = mice with range > 5 %', fontsize=10.5)
    save_fig(fig, OUT_DIR, 'lambda_inertness_heatmaps')


# ----------------------------------------------------------------------
# prior scoring
# ----------------------------------------------------------------------
def _signs(diff):
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    return f'{pos}+/{neg}-/{diff.size - pos - neg}0 of {diff.size}'


def score_priors(kl, sh, rr_kl, rr_sh, cl):
    L = {l: i for i, l in enumerate(LOSS_ORDER)}
    A = {a: j for j, a in enumerate(ARCH_ORDER)}
    print('\n=== PRIOR (e) SCORING — temporal decoder, 6 mice ===')
    print(f'\n(e1) lambda inert for calibrated losses at ALL widths '
          f'(median-over-mice worst relative range of KL skill <= {100 * INERT_THR:.0f}%)')
    print(f'{"loss":10s}' + ''.join(f'{a:>9s}' for a in ARCH_ORDER) + '   groups inert')
    verdict = {}
    for lf in CALIBRATED:
        i = L[lf]
        med = np.median(rr_kl[i], axis=1)
        ok = med <= INERT_THR
        verdict[lf] = int(ok.sum())
        print(f'{lf:10s}' + ''.join(f'{100 * v:8.1f}%' for v in med)
              + f'   {int(ok.sum())}/{len(ARCH_ORDER)}')
    n_ok = sum(verdict[l] for l in ('kl', 'js', 'ce'))
    print(f'  -> kl/js/ce: {n_ok}/21 (loss, arch) groups inert  '
          f'[{"SUPPORTED" if n_ok == 21 else "NOT supported"}]')
    print(f'  -> pcaflat : {verdict["pcaflat"]}/7 groups inert  '
          f'[{"SUPPORTED" if verdict["pcaflat"] == 7 else "FALSIFIED"}]')
    # pcaflat direction: is the degradation monotone with lambda?
    i = L['pcaflat']
    d_end = kl[i, :, :, -1] - kl[i, :, :, 0]
    rho = np.array([[spearmanr(np.arange(5), kl[i, j, k])[0] for k in MICE]
                    for j in range(len(ARCH_ORDER))])
    print('  pcaflat KL skill, lambda 3e-3 minus 0, sign count per arch:')
    for j, ar in enumerate(ARCH_ORDER):
        print(f'     {ar:5s} {_signs(d_end[j]):>16s}   median Δ = {np.median(d_end[j]):+.2f}'
              f'   Spearman(λ rank, skill) = +1 in {int((rho[j] > 0.999).sum())}/6 mice'
              f'   skill<1 at λ=0: {int((kl[i, j, :, 0] < 1).sum())}/6, at λ=3e-3: '
              f'{int((kl[i, j, :, -1] < 1).sum())}/6')
    print(f'  pcaflat s_hat 3e-3 minus 0: {_signs((sh[i, :, :, -1] - sh[i, :, :, 0]).ravel())} '
          f'(arch x mouse); median s_hat λ=0 {np.median(sh[i, :, :, 0]):.2f} -> '
          f'λ=3e-3 {np.median(sh[i, :, :, -1]):.2f}')
    # calibrated: the largest single-mouse range anywhere
    for lf in ('kl', 'js', 'ce'):
        i = L[lf]
        jj, kk = np.unravel_index(np.argmax(rr_kl[i]), rr_kl[i].shape)
        print(f'  {lf}: largest single-mouse KL-skill range = {100 * rr_kl[i, jj, kk]:.2f}% '
              f'({ARCH_ORDER[jj]}, mouse {kk}); largest s_hat range = '
              f'{100 * rr_sh[i].max():.2f}%')
    print(f'  kl vs ce (identical gradients; CE = KL + H(target)): max |Δ KL skill| = '
          f'{np.abs(kl[L["kl"]] - kl[L["ce"]]).max():.1e}')

    # ---- (e2) evar instability grows with lambda AND width
    i = L['pca']
    print('\n(e2) evar instability grows with lambda AND width (temporal, projection-evar)')
    print('  "with lambda": KL skill at λ=3e-3 minus λ=0, and Spearman(λ rank, skill) > 0, '
          'per mouse')
    for j, ar in enumerate(ARCH_ORDER):
        d = kl[i, j, :, -1] - kl[i, j, :, 0]
        r = np.array([spearmanr(np.arange(5), kl[i, j, k])[0] for k in MICE])
        ds = sh[i, j, :, -1] - sh[i, j, :, 0]
        print(f'     {ar:5s} KL skill Δ {_signs(d):>16s}  median Δ {np.median(d):+.2f} '
              f'(λ=0 median {np.median(kl[i, j, :, 0]):.2f}); rho>0 in '
              f'{int((r > 0).sum())}/6; s_hat Δ {_signs(ds):>16s}')
    d_all = (kl[i, :, :, -1] - kl[i, :, :, 0]).ravel()
    print(f'     all arch x mouse: KL skill worse at λ=3e-3 than λ=0 in {_signs(d_all)}')
    # monotone? count of mice where the worst lambda is the largest lambda
    worst_is_max = (kl[i].argmax(-1) == 4)
    print(f'     worst KL skill sits at λ=3e-3 in {int(worst_is_max.sum())}/42 arch x mouse; '
          f'at λ=0 in {int((kl[i].argmax(-1) == 0).sum())}/42')
    print('  "with width": relative range over λ (the heatmap quantity) per mouse, '
          'h64 vs h8, h64 vs h4, Spearman(width, range) over h4..h64')
    for name, rr in (('KL skill', rr_kl), ('s_hat', rr_sh)):
        r64_8 = rr[i, A['h64']] - rr[i, A['h8']]
        r64_4 = rr[i, A['h64']] - rr[i, A['h4']]
        wj = [A[a] for a in WIDTH_ARCHS]
        rho_w = np.array([spearmanr(np.arange(5), rr[i, wj, k])[0] for k in MICE])
        print(f'     {name:9s} range(h64)-range(h8): {_signs(r64_8):>16s} '
              f'(medians h8 {100 * np.median(rr[i, A["h8"]]):.0f}% / h64 '
              f'{100 * np.median(rr[i, A["h64"]]):.0f}%);  '
              f'range(h64)-range(h4): {_signs(r64_4):>16s};  '
              f'rho(width, range) > 0 in {int((rho_w > 0).sum())}/6 mice '
              f'(median rho {np.median(rho_w):+.2f})')
        medr = [100 * np.median(rr[i, A[a]]) for a in ARCH_ORDER]
        print('               median range by arch: '
              + '  '.join(f'{a}={v:.0f}%' for a, v in zip(ARCH_ORDER, medr)))
    # interaction: lambda effect at h64 vs h8 per mouse (the AND)
    d8 = kl[i, A['h8'], :, -1] - kl[i, A['h8'], :, 0]
    d64 = kl[i, A['h64'], :, -1] - kl[i, A['h64'], :, 0]
    print(f'  AND: (Δ skill over λ at h64) − (Δ at h8) per mouse: {_signs(d64 - d8)}; '
          f'median Δ h8 {np.median(d8):+.2f}, h64 {np.median(d64):+.2f}')
    # s_hat clamping for evar — a clamped ladder end (6.0) caps the range, so the
    # s_hat width comparison is a BOUND where clamping bites
    print('  evar s_hat clamped cells (ladder end 6.0 = bound; caps the range) per arch: '
          + '  '.join(f'{a}={int(cl[i, A[a]].sum())}/30' for a in ARCH_ORDER)
          + f'; mice with >=1 clamp at h64: {int((cl[i, A["h64"]].sum(-1) > 0).sum())}/6, '
          f'at h8: {int((cl[i, A["h8"]].sum(-1) > 0).sum())}/6')


def main():
    t = load()
    kl = grid(t, 'kl_skill')
    sh = grid(t, 's_hat')
    cl = grid(t, 's_hat_clamped')
    ag = grid(t, 's_hat_agreement')
    n_cl = int(cl.sum())
    n_ag = int((ag > 0.10).sum())
    cl_by_loss = {lf: int(cl[i].sum()) for i, lf in enumerate(LOSS_ORDER)}
    print(f'temporal cells: {kl.size}; s_hat clamped in {n_cl} ({cl_by_loss}); '
          f's_hat_agreement > 0.10 (decoder reshapes; s_hat under-describes) in {n_ag}/{ag.size}')

    rr_kl, rr_sh = rel_range(kl), rel_range(sh)          # [loss, arch, mouse]
    med_kl, med_sh = np.median(rr_kl, -1), np.median(rr_sh, -1)
    n_kl, n_sh = (rr_kl > INERT_THR).sum(-1), (rr_sh > INERT_THR).sum(-1)

    fig_grid(kl, 'lambda_kl_skill_grid',
             'KL skill\nKL(tgt‖dec) / KL(tgt‖LOO mean)',
             'Temporal decoder: KL skill vs entropy weight λ_H (held-out trials; 1 = '
             'LOO predict-mean null, < 1 beats it). Rows: training loss; columns: '
             'architecture.\nSpatial decoder not shown — λ_H does not enter it '
             '(its 5 λ cells per group are bit-identical replicates). y per row: '
             'log where range > 4×.')
    fig_grid(sh, 'lambda_s_hat_grid',
             's_hat\n(equivalent sharpening)',
             'Temporal decoder: equivalent sharpening ŝ vs entropy weight λ_H '
             '(1 = none; calibration curve per mouse from the targets). '
             f'Clamped inversions: {n_cl}/{cl.size} (hollow); '
             f'agreement > 0.10 (reshape, ŝ under-describes): {n_ag}/{ag.size}.\n'
             'Spatial decoder not shown — λ_H does not enter it (exact replicates). '
             'y per row: log where range > 4×.',
             clamped=cl)
    fig_heatmaps(med_kl, med_sh, n_kl, n_sh)

    # summary csv for the log
    rows = []
    for i, lf in enumerate(LOSS_ORDER):
        for j, ar in enumerate(ARCH_ORDER):
            rows.append(dict(loss_family=lf, arch=ar,
                             kl_skill_range_median=med_kl[i, j],
                             kl_skill_range_n_over_5pct=int(n_kl[i, j]),
                             s_hat_range_median=med_sh[i, j],
                             s_hat_range_n_over_5pct=int(n_sh[i, j]),
                             kl_skill_lam0_median=float(np.median(kl[i, j, :, 0])),
                             kl_skill_lam3e3_median=float(np.median(kl[i, j, :, -1])),
                             s_hat_lam0_median=float(np.median(sh[i, j, :, 0])),
                             s_hat_lam3e3_median=float(np.median(sh[i, j, :, -1])),
                             n_clamped=int(cl[i, j].sum())))
    pd.DataFrame(rows).to_csv(OUT_DIR / 'lambda_inertness.csv', index=False)
    print(f'  -> lambda_inertness.csv ({len(rows)} rows)')

    score_priors(kl, sh, rr_kl, rr_sh, cl)


if __name__ == '__main__':
    main()
