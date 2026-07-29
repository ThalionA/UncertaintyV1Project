# -*- coding: utf-8 -*-
"""Characterise the decoders trained at the selected hyperparameters (Task 2, steps B & C).

Chosen cell (see `hp_best_select.py` for the rule and the ranking):
    hpsweep_v2 / lam0p003_drop0p75_acttanh_h8_pat10_vf0p2_wd0p0001_shp0 / Q_KL_half_100ms
    loss = KL, dropout 0.75, patience 10, H = 8, tanh, entropy λ_H = 0.003,
    weight decay 1e-4, val fraction 0.2, shape λ = 0.

STEP B  fig6 — example decoded posteriors vs the IO target, spatial and temporal
        overlaid on the same trials. Trials are picked by the PRESENTED STIMULUS
        ORIENTATION only (evenly spaced distinct orientation levels, lowest trial index
        within each), so the gallery is a tour of the stimulus range and is NOT selected
        on peakiness, target width, decoder error, dispersion or contrast.

STEP C  fig7 — spatial vs temporal head-to-head at that cell, three ways:
        (a) ACROSS animals   — paired over the 6 mice on KL normalised loss (LOO
            predict-mean null), also with mouse_2 dropped (n = 5).
        (b) WITHIN each animal — paired over that animal's held-out trials
            (n = 326-470) using `nn_classifier.fit_loss_per_trial`. Both decoders are
            scored on the SAME test trials with the SAME targets (verified), so the
            trials pair one-to-one. This is within-animal RELIABILITY, not population
            evidence: n is trials, and trials within an animal are not independent
            replicates of the biological effect.
        (c) HIERARCHICAL — mixed-effects model, per-trial loss ~ architecture with a
            random intercept per mouse (and, as a robustness check, a random slope for
            architecture too). The random effect absorbs the between-animal offsets, so
            the architecture coefficient is estimated from WITHIN-animal contrasts
            instead of treating 2186 trials as 2186 independent animals.

Outputs (PNG+SVG) under figures/hparam_summary/.
Usage:  python diagnostics/hp_best_characterise.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import peakiness_style as ps  # noqa: E402
from nn_classifier import fit_loss_per_trial  # noqa: E402
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

RUN = 'hpsweep_v2'
CELL = 'lam0p003_drop0p75_acttanh_h8_pat10_vf0p2_wd0p0001_shp0'
SLUG = 'Q_KL_half_100ms'
METRIC = 'KL'                      # the calibrated scoring metric
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
ANOMALOUS = 'mouse_2'


def load(results_root, run=RUN, cell=CELL, slug=SLUG):
    p = Path(results_root) / run / cell / slug / 'stratified_balanced.mat'
    if not p.is_file():
        raise SystemExit(f'missing {p}')
    return sio.loadmat(str(p), simplify_cells=True)['results']


# ------------------------------------------------------------------ per-trial losses
def per_trial(res, mouse):
    """(loss_spat, loss_temp, null_mean) for one mouse — same trials, same targets."""
    D = res[mouse]['Dist']
    tgt = np.asarray(D['spat']['target'], float)
    tgt_t = np.asarray(D['temp']['target'], float)
    assert tgt.shape == tgt_t.shape and np.allclose(tgt, tgt_t, atol=1e-9), \
        f'{mouse}: spatial and temporal targets differ — trials do not pair'
    ok = np.isfinite(tgt).all(1)
    for a, _ in ARCHS:
        ok &= np.isfinite(np.asarray(D[a]['decoded'], float)).all(1)
    T = torch.tensor(tgt[ok])
    out = []
    for a, _ in ARCHS:
        P = torch.tensor(np.asarray(D[a]['decoded'], float)[ok])
        out.append(fit_loss_per_trial(P, T, METRIC).numpy().astype(float))
    # LOO predict-mean null on the same trials (identical construction to
    # performance_vs_hparams._norm_by_mouse), reduced to its mean for scaling
    n = int(ok.sum())
    tot = tgt[ok].sum(0)
    pm = (tot[None, :] - tgt[ok]) / (n - 1)
    null = fit_loss_per_trial(torch.tensor(pm), T, METRIC).numpy().astype(float)
    return out[0], out[1], float(null.mean())


# ------------------------------------------------------------------ STEP B
def fig_examples(res, out_root, n=6, mouse='mouse_0', label='KL, dropout 0.75, patience 10',
                 stem='hp_fig6_best_example_posteriors'):
    ps.apply()
    D = res[mouse]['Dist']
    tgt = np.asarray(D['spat']['target'], float)
    dec = {a: np.asarray(D[a]['decoded'], float) for a, _ in ARCHS}
    ori = np.asarray(res[mouse]['trials']['orientation'], float)
    disp = np.asarray(res[mouse]['trials']['dispersion'], float)
    con = np.asarray(res[mouse]['trials']['contrast'], float)
    ok = np.isfinite(tgt).all(1)
    for a in dec:
        ok &= np.isfinite(dec[a]).all(1)
    idx = np.flatnonzero(ok)
    # SELECTION: the presented STIMULUS orientation only. Take the distinct orientation
    # levels the animal was shown, drop 90 deg as the wrapped duplicate of 0 deg, keep n
    # evenly spaced levels, and inside each level take the LOWEST TRIAL INDEX — arbitrary
    # with respect to stimulus dispersion and contrast. Nothing about the target's width,
    # the decoded posteriors, their peakiness or their error enters this choice.
    # (Two rejected alternatives: sorting by the target's ARGMAX — the orientation axis
    # wraps, so argmax piles up at both ends and the "spread" is an artefact; and even
    # quantiles of the trial list — that just reproduces the design's trial counts, which
    # here put 4 of 6 examples at 0/90 deg, i.e. the same orientation.)
    levels = np.unique(ori[idx])
    if 0.0 in levels and 90.0 in levels:
        levels = levels[levels != 90.0]
    levels = levels[np.linspace(0, len(levels) - 1, n).astype(int)]
    picks = np.array([idx[ori[idx] == lv][0] for lv in levels])

    ncol, nrow = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=ps.figsize(ncol, nrow),
                             squeeze=False, sharex=True, sharey=True)
    x = np.arange(tgt.shape[1])
    ymax = 1.15 * max(float(tgt[picks].max()),
                      max(float(dec[a][picks].max()) for a in dec))
    for k, tr in enumerate(picks):
        ax = axes[k // ncol][k % ncol]
        ps.target_band(ax, x, tgt[tr])
        for a, alab in ARCHS:
            ax.plot(x, dec[a][tr], color=ps.ARCH[a], lw=1.7, label=f'{alab} decoded')
        ax.set_ylim(0, ymax)
        ax.set_title(f'trial {tr}:  ori {ori[tr]:.0f}°, disp {disp[tr]:.0f}°, '
                     f'contrast {con[tr]:g}', fontsize=8.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel('orientation (deg)')
        if k % ncol == 0:
            ax.set_ylabel('probability')
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].legend(fontsize=7, frameon=True, loc='upper center')
    ps.label_panels(axes.ravel()[:n])
    fig.suptitle(f'Decoded and IO target posteriors — {label} '
                 f'({mouse}, {len(idx)} held-out trials)', y=1.01, fontsize=10)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), stem)
    print(f'  picked trials {list(map(int, picks))}')
    print(f'  stimulus orientation {[float(ori[t]) for t in picks]} deg')
    print(f'  stimulus dispersion  {[float(disp[t]) for t in picks]} deg   '
          f'contrast {[float(con[t]) for t in picks]}')
    print(f'  target max-prob {[round(float(tgt[t].max()), 4) for t in picks]}')
    for a, alab in ARCHS:
        print(f'  {alab:>8} max-prob {[round(float(dec[a][t].max()), 4) for t in picks]}')


# ------------------------------------------------------------------ STEP C
def stats_and_fig(res, out_root, results_root, label='KL, dropout 0.75, patience 10',
                  stem='hp_fig7_best_spat_vs_temp'):
    mice = sorted(m for m in res if isinstance(res[m], dict) and 'Dist' in res[m])
    print(f'\n=== STEP C — spatial vs temporal at the chosen cell ({len(mice)} mice) ===')

    # ---- (i) across animals -------------------------------------------------
    nlm = {a: np.array(_norm_by_mouse(res, a)[(METRIC, 'pm')], float) for a, _ in ARCHS}
    per_mouse = {a: {} for a, _ in ARCHS}
    for a, _ in ARCHS:
        for m, v in zip(mice, nlm[a]):
            per_mouse[a][m] = v
    keep = {'all (n=6)': mice, f'excl. {ANOMALOUS} (n=5)': [m for m in mice if m != ANOMALOUS]}
    across = {}
    for lab, ms in keep.items():
        s = np.array([per_mouse['spat'][m] for m in ms])
        t = np.array([per_mouse['temp'][m] for m in ms])
        tt = stats.ttest_rel(s, t)
        d = s - t
        across[lab] = dict(s=s, t=t, tstat=float(tt.statistic), p=float(tt.pvalue),
                           dz=float(d.mean() / d.std(ddof=1)), n=len(ms),
                           fav_spat=int((s < t).sum()))
        print(f'\n(i) ACROSS animals, {lab}:')
        print(f'    spatial  {s.mean():.4f} +/- {s.std(ddof=1)/np.sqrt(len(s)):.4f}')
        print(f'    temporal {t.mean():.4f} +/- {t.std(ddof=1)/np.sqrt(len(t)):.4f}')
        print(f'    paired t({len(ms)-1}) = {tt.statistic:.3f}, p = {tt.pvalue:.4f}, '
              f"dz = {across[lab]['dz']:.3f}")
        print(f'    mice favouring spatial (lower loss): {across[lab]["fav_spat"]}/{len(ms)}; '
              f'temporal: {len(ms)-across[lab]["fav_spat"]}/{len(ms)}')
        for m in ms:
            print(f'       {m}: spat {per_mouse["spat"][m]:.4f}  temp {per_mouse["temp"][m]:.4f}  '
                  f'diff(t-s) {per_mouse["temp"][m]-per_mouse["spat"][m]:+.4f}')

    # ---- (ii) within animal --------------------------------------------------
    print('\n(ii) WITHIN animal — paired over that animal\'s held-out trials.')
    print('     NOTE: n is TRIALS. These p-values are within-animal reliability, NOT')
    print('     population evidence — trials inside one animal are not independent')
    print('     replicates of the biological effect.')
    W, rowsdf = {}, []
    for m in mice:
        ls, lt, null = per_trial(res, m)
        d = lt - ls                       # >0 => temporal worse
        tt = stats.ttest_rel(lt, ls)
        W[m] = dict(n=len(d), mean=float(d.mean()), sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                    dz=float(d.mean() / d.std(ddof=1)), p=float(tt.pvalue),
                    t=float(tt.statistic), pct_spat=100.0 * float((ls < lt).mean()),
                    null=null,
                    mean_n=float((d / null).mean()),
                    sem_n=float((d / null).std(ddof=1) / np.sqrt(len(d))))
        print(f'    {m}: n={W[m]["n"]:4d}  mean(temp-spat) = {W[m]["mean"]:+.4f} '
              f'+/- {W[m]["sem"]:.4f}   dz = {W[m]["dz"]:+.3f}  t({len(d)-1}) = {W[m]["t"]:+.2f}  '
              f'p = {W[m]["p"]:.3g}   trials favouring spatial: {W[m]["pct_spat"]:.1f}%')
        for a, _ in ARCHS:
            v = ls if a == 'spat' else lt
            rowsdf.append(pd.DataFrame({'loss': v, 'loss_n': v / null,
                                        'arch': a, 'mouse': m}))
    df = pd.concat(rowsdf, ignore_index=True)
    df['arch'] = pd.Categorical(df['arch'], categories=['spat', 'temp'])

    # ---- (iii) hierarchical --------------------------------------------------
    import statsmodels.formula.api as smf
    print('\n(iii) HIERARCHICAL mixed-effects model')
    print(f'      rows = {len(df)} ({len(df)//2} trials x 2 architectures), groups = mouse')
    fits = {}
    for lab, formula, re_f, dv in [
            ('random intercept (primary)', 'loss ~ arch', None, 'loss'),
            ('random intercept + random slope', 'loss ~ arch', '~arch', 'loss'),
            ('random intercept, per-mouse normalised loss', 'loss_n ~ arch', None, 'loss_n')]:
        md = smf.mixedlm(formula, df, groups=df['mouse'], re_formula=re_f)
        r = md.fit(reml=True, method='lbfgs')
        name = [n for n in r.params.index if n.startswith('arch')][0]
        beta, se = float(r.params[name]), float(r.bse[name])
        z, p = float(r.tvalues[name]), float(r.pvalues[name])
        gv = float(r.cov_re.iloc[0, 0])
        fits[lab] = dict(beta=beta, se=se, z=z, p=p, gv=gv, resid=float(r.scale))
        print(f'    [{lab}]  formula: {formula}, groups=mouse'
              + (f', re_formula={re_f}' if re_f else ''))
        print(f'        fixed effect {name} (temporal - spatial) = {beta:+.5f}  '
              f'SE {se:.5f}  z = {z:+.3f}  p = {p:.3g}')
        print(f'        random-intercept variance = {gv:.5f}  '
              f'(sd {np.sqrt(max(gv,0)):.4f}), residual variance = {r.scale:.5f}')
        if re_f:
            print(f'        random-slope variance   = {float(r.cov_re.iloc[1,1]):.5f}')

    # ---- figure --------------------------------------------------------------
    ps.apply()
    fig, ax = plt.subplots(1, 3, figsize=ps.figsize(3, 1))

    # (a) across animals
    a0 = ax[0]
    for m in mice:
        st = ANOMALOUS == m
        a0.plot([0, 1], [per_mouse['spat'][m], per_mouse['temp'][m]],
                color='0.6', lw=1.1, ls='--' if st else '-', zorder=1,
                label=(f'{ANOMALOUS} (flagged)' if st else ('individual mice' if m == mice[0] else None)))
    for j, (a, alab) in enumerate(ARCHS):
        v = np.array([per_mouse[a][m] for m in mice])
        a0.errorbar([j], [v.mean()], yerr=[v.std(ddof=1) / np.sqrt(len(v))], fmt='o',
                    ms=9, color=ps.ARCH[a], capsize=4, lw=2, zorder=3,
                    label=f'{alab} mean ± SEM (n=6)')
    a0.axhline(1.0, color='0.45', ls=':', lw=1.2, label='chance')
    a0.set_xlim(-0.4, 1.4)
    a0.set_xticks([0, 1]); a0.set_xticklabels([a[1] for a in ARCHS])
    a0.set_xlabel('architecture')
    a0.set_ylabel('KL normalised loss (÷ predict-mean)')
    a0.set_title('across animals', fontsize=10)
    a0.legend(fontsize=6.5, frameon=True, loc='best')

    # (b) within animal — % of held-out trials favouring spatial
    a1 = ax[1]
    xs = np.arange(len(mice))
    pct = [W[m]['pct_spat'] for m in mice]
    a1.bar(xs, pct, 0.62, color=[ps.ARCH['spat'] if p > 50 else ps.ARCH['temp'] for p in pct],
           edgecolor='k', lw=0.5)
    a1.axhline(50, color='k', ls=':', lw=1.3, label='no preference (50%)')
    a1.set_xticks(xs); a1.set_xticklabels([m.replace('mouse_', 'm') for m in mice])
    a1.set_xlabel('mouse')
    a1.set_ylabel('held-out trials favouring spatial (%)')
    a1.set_ylim(0, 100)
    a1.set_title('within animals (n = trials)', fontsize=10)
    a1.legend(handles=[Line2D([0], [0], color='k', ls=':', lw=1.3, label='no preference (50%)'),
                       Line2D([0], [0], marker='s', ls='', color=ps.ARCH['spat'], label='spatial favoured'),
                       Line2D([0], [0], marker='s', ls='', color=ps.ARCH['temp'], label='temporal favoured')],
               fontsize=6.5, frameon=True, loc='best')

    # (c) hierarchical forest — per-mouse within-animal effect + pooled fixed effect
    a2 = ax[2]
    ys = np.arange(len(mice))
    a2.errorbar([W[m]['mean'] for m in mice], ys, xerr=[W[m]['sem'] for m in mice],
                fmt='o', ms=5, color='0.35', capsize=3, lw=1.2,
                label='per-mouse mean ± SEM over trials')
    f = fits['random intercept (primary)']
    f2 = fits['random intercept + random slope']
    a2.errorbar([f['beta']], [len(mice) + 0.4], xerr=[f['se']], fmt='D', ms=8,
                color=ps.KL, capsize=4, lw=2, label='mixedlm fixed effect ± SE\n(random intercept)')
    a2.errorbar([f2['beta']], [len(mice) + 1.2], xerr=[f2['se']], fmt='s', ms=8,
                color=ps.JS, capsize=4, lw=2, label='mixedlm fixed effect ± SE\n(+ random slope)')
    a2.axvline(0.0, color='k', ls=':', lw=1.3, label='no difference')
    a2.set_yticks(list(ys) + [len(mice) + 0.4, len(mice) + 1.2])
    a2.set_yticklabels([m.replace('mouse_', 'm') for m in mice] + ['pooled (RI)', 'pooled (RI+RS)'])
    a2.set_ylabel('mouse / pooled estimate')
    a2.set_xlabel('per-trial KL loss, temporal − spatial')
    a2.set_title('hierarchical', fontsize=10)
    a2.legend(fontsize=6, frameon=True, loc='lower left')

    ps.label_panels(ax)
    fig.suptitle('Spatial vs temporal decoder at the selected hyperparameters '
                 f'({label})', y=1.02, fontsize=10)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), stem)
    return across, W, fits


def main(results_root, out_root, mouse, n, cell, slug, label, stem_b, stem_c):
    res = load(results_root, cell=cell, slug=slug)
    print(f'=== cell: {cell} / {slug}  ({label}) ===')
    print('=== STEP B — example posteriors ===')
    fig_examples(res, out_root, n=n, mouse=mouse, label=label, stem=stem_b)
    stats_and_fig(res, out_root, results_root, label=label, stem=stem_c)
    print(f'\nDone -> {Path(out_root).resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/hparam_summary')
    ap.add_argument('--mouse', default='mouse_0')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--cell', default=CELL)
    ap.add_argument('--slug', default=SLUG)
    ap.add_argument('--label', default='KL, dropout 0.75, patience 10')
    ap.add_argument('--stem-b', default='hp_fig6_best_example_posteriors')
    ap.add_argument('--stem-c', default='hp_fig7_best_spat_vs_temp')
    a = ap.parse_args()
    main(a.results_root, a.out_root, a.mouse, a.n, a.cell, a.slug, a.label,
         a.stem_b, a.stem_c)
