# -*- coding: utf-8 -*-
"""Compare three PCA-loss variants on real data (2026-06-03, Theo's request).

Variants (fresh matched run `wm3`, Q / half / 100 ms / stratified_balanced / 6 mice):
  - PCA (evar-weighted)      : the standard weighted loss (peaky)
  - flat-evar (Brier)        : uniform weights (slightly broad)
  - PCA + shape term (λ=1)   : evar floored by λ/100 (width-matched fix)

All three are PCA loss with the same split seed, so trials are aligned. We score
the held-out decoded posteriors of each variant under a COMMON metric (true-evar
PCA loss, and KL — both blind to which variant trained the net), normalised to the
per-arch shuffle control saved in each .mat, and compare:
  - spatial vs temporal, raw and shuffle-normalised (skill = loss/shuffle);
  - across mice (mean ± sem, paired t at n=6 — Wilcoxon avoided per GOTCHAS) and
    within mice (per-mouse paired);
  - decoded peakiness (max-prob) vs the IO target;
  - example posteriors.

Outputs under figures/loss_variants/:
  examples_<arch>.png        target vs the 3 variants, matched trials
  peakiness.png              decoded max-prob per variant/arch (across + per mouse)
  spat_vs_temp_<metric>.png  raw + skill, grouped by variant, paired-t annotated
  within_mice_<metric>.png   per-mouse spat-vs-temp, faceted by variant
  summary.csv                every (variant, arch, metric): real, shuffle, skill, p

Usage:  python compare_loss_variants.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import peakiness_style as ps
from cross_loss_eval import _eval_one

SLUG = 'Q_PCA_half_100ms_all'
# Auto-discover variants present under results/: the evar default (wm3), the
# flat-evar control, and every shape-λ run (wm3_shape<λ>). Greens for the shape
# sweep, darkening with λ.
_SHAPE_GREENS = ps.SHAPE_GREENS


def discover_variants(results_root):
    out = [('PCA (evar)', 'wm3'), ('flat-evar', 'wm3_flatevar')]
    shapes = sorted(Path(results_root).glob('wm3_shape*'),
                    key=lambda p: float(p.name.replace('wm3_shape', '').replace('p', '.')))
    for p in shapes:
        lam = p.name.replace('wm3_shape', '').replace('p', '.')
        out.append((f'PCA+shape λ={lam}', p.name))
    return out


def variant_colors(variants):
    col = {'PCA (evar)': ps.PCA_EVAR, 'flat-evar': ps.FLAT_EVAR}
    gi = 0
    for name, _ in variants:
        if name.startswith('PCA+shape'):
            col[name] = _SHAPE_GREENS[min(gi, len(_SHAPE_GREENS) - 1)]; gi += 1
    return col


VARIANTS = []   # set in main() from discover_variants
VCOL = {}       # set in main() from variant_colors
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
METRICS = ['PCA', 'KL']


def _H(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return -np.sum(p * np.log(p), -1)


def load_variant(results_root, run, split):
    mat = Path(results_root) / run / SLUG / f'{split}.mat'
    if not mat.is_file():
        return None
    return sio.loadmat(str(mat), simplify_cells=True)['results']


def common_basis(res_evar):
    """Per-mouse (pcs, true_evar) from the evar variant — the common PCA metric."""
    basis = {}
    for mk, m in res_evar.items():
        basis[mk] = (np.asarray(m['Dist']['pcs'], float),
                     np.asarray(m['Dist']['explained_var'], float))
    return basis


def collect(results_root, split):
    res = {name: load_variant(results_root, run, split) for name, run in VARIANTS}
    res = {k: v for k, v in res.items() if v is not None}
    if 'PCA (evar)' not in res:
        raise SystemExit('evar variant (wm3) not found — run it first.')
    basis = common_basis(res['PCA (evar)'])
    mice = sorted(res['PCA (evar)'].keys())
    # data[variant][arch][metric] = {'real':[per mouse], 'shuf':[...], 'skill':[...]}
    data, peaki = {}, {}
    for name, r in res.items():
        data[name] = {a: {met: {'real': [], 'shuf': [], 'skill': []} for met in METRICS}
                      for a, _ in ARCHS}
        peaki[name] = {a: {'maxprob': [], 'H': []} for a, _ in ARCHS}
        for mk in mice:
            if mk not in r:
                continue
            pcs, evar = basis[mk]
            for a, _ in ARCHS:
                D = r[mk]['Dist']
                dec = np.asarray(D[a]['decoded'], float)
                tgt = np.asarray(D[a]['target'], float)
                shf = np.asarray(D[a + '_shf']['decoded'], float)
                peaki[name][a]['maxprob'].append(dec.max(1).mean())
                peaki[name][a]['H'].append(_H(dec).mean())
                for met in METRICS:
                    real = _eval_one(dec, tgt, met, pcs, evar)
                    shuf = _eval_one(shf, tgt, met, pcs, evar)
                    data[name][a][met]['real'].append(real)
                    data[name][a][met]['shuf'].append(shuf)
                    data[name][a][met]['skill'].append(real / shuf if shuf else np.nan)
    tgt_mp = np.mean([np.asarray(res['PCA (evar)'][mk]['Dist']['spat']['target'], float).max(1).mean()
                      for mk in mice])
    return res, data, peaki, mice, tgt_mp


# ----------------------------------------------------------------------

def fig_examples(res, arch, label, out_dir, mouse='mouse_0', n=4):
    tgt = np.asarray(res['PCA (evar)'][mouse]['Dist'][arch]['target'], float)
    peak = np.argmax(tgt, 1)
    order = np.argsort(peak)
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    ymax = 4.0 * float(np.median(tgt[picks].max(1)))
    # show evar, flat, and the shape variant whose peakiness best matches the
    # target (the recommended λ) — keep it legible
    tgt_mp = tgt.max(1).mean()
    shapes = [k for k in res if k.startswith('PCA+shape')]
    best_shape = min(shapes, key=lambda k: abs(
        np.asarray(res[k][mouse]['Dist'][arch]['decoded'], float).max(1).mean() - tgt_mp)) \
        if shapes else None
    show = ['PCA (evar)', 'flat-evar'] + ([best_shape] if best_shape else [])
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.8), sharey=True)
    x = np.arange(tgt.shape[1])
    for ax, tr in zip(axes, picks):
        ax.fill_between(x, tgt[tr], color='0.75', alpha=0.8, lw=0, label='IO target')
        for name in show:
            dec = np.asarray(res[name][mouse]['Dist'][arch]['decoded'], float)
            ax.plot(x, dec[tr], color=VCOL[name], lw=1.7, label=name)
        ax.set_ylim(0, ymax); ax.set_yticks([]); ax.set_xlabel('orientation bin')
        ax.set_title(f'trial {tr}', fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f'Example posteriors — {label} ({arch}), {mouse}; y capped, spikes clip', y=1.04)
    _save(fig, out_dir, f'examples_{arch}')


def _lambda_order(names):
    """Order variants as a λ progression: evar (λ=0), shapes by ascending λ, then
    flat-evar (λ=∞). Returns (ordered_names, xtick_labels)."""
    evar = [n for n in names if n == 'PCA (evar)']
    flat = [n for n in names if n == 'flat-evar']
    shapes = sorted([n for n in names if n.startswith('PCA+shape')],
                    key=lambda n: float(n.split('λ=')[1]))
    ordered = evar + shapes + flat
    labels = []
    for n in ordered:
        if n == 'PCA (evar)':
            labels.append('evar\n(λ=0)')
        elif n == 'flat-evar':
            labels.append('flat\n(λ=∞)')
        else:
            labels.append('λ=' + n.split('λ=')[1])
    return ordered, labels


def fig_examples_grid(res, arch, label, out_dir, mouse='mouse_0', n=4):
    """One ROW per variant (λ progression), columns = example trials — see each
    variant's fitted posteriors separately rather than overlaid."""
    names, _ = _lambda_order(list(res.keys()))
    tgt = np.asarray(res['PCA (evar)'][mouse]['Dist'][arch]['target'], float)
    order = np.argsort(np.argmax(tgt, 1))
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    ymax = 4.0 * float(np.median(tgt[picks].max(1)))
    x = np.arange(tgt.shape[1])
    fig, axes = plt.subplots(len(names), n, figsize=(3.0 * n, 1.9 * len(names)),
                             squeeze=False, sharex=True, sharey=True)
    for r, name in enumerate(names):
        dec = np.asarray(res[name][mouse]['Dist'][arch]['decoded'], float)
        for c, tr in enumerate(picks):
            ax = axes[r][c]
            ax.fill_between(x, tgt[tr], color='0.78', alpha=0.8, lw=0)
            ax.plot(x, dec[tr], color=VCOL[name], lw=1.6)
            ax.set_ylim(0, ymax); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'trial {tr}', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{name}\nmp={dec[tr].max():.2f}', fontsize=7.5,
                              color=VCOL[name])
            if r == len(names) - 1:
                ax.set_xlabel('orientation bin', fontsize=8)
    fig.suptitle(f'Fitted posteriors by variant — {label} ({arch}), {mouse}; '
                 'grey = IO target (y capped, spikes clip)', y=1.005, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f'examples_grid_{arch}')


def fig_metric_vs_lambda(data, out_dir):
    """KL vs PCA skill across the λ progression, spat & temp separately, plus the
    KL/PCA ratio (the 'calibration penalty' the PCA metric hides)."""
    names, labels = _lambda_order(list(data.keys()))
    xs = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, (a, alab) in zip(axes[:2], ARCHS):
        for met, mc, mk in (('PCA', '#999999', 'o'), ('KL', '#7b3294', 's')):
            y = [np.nanmean(data[n][a][met]['skill']) for n in names]
            e = [np.nanstd(data[n][a][met]['skill'], ddof=1) /
                 np.sqrt(np.isfinite(data[n][a][met]['skill']).sum()) for n in names]
            ax.errorbar(xs, y, yerr=e, color=mc, marker=mk, lw=2, capsize=3,
                        label=f'{met} skill')
        ax.axhline(1.0, ls=':', color='r', lw=1, label='chance')
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('skill = loss / shuffle  (<1 beats chance)')
        ax.set_title(f'{alab} ({a})'); ax.legend(frameon=False, fontsize=8)
    # ratio panel
    ax = axes[2]
    for a, alab in ARCHS:
        ratio = [np.nanmean(data[n][a]['KL']['skill']) / np.nanmean(data[n][a]['PCA']['skill'])
                 for n in names]
        ax.plot(xs, ratio, marker='o', lw=2, label=alab)
    ax.axhline(1.0, ls=':', color='0.5', lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('KL-skill / PCA-skill  (calibration penalty)')
    ax.set_title('How much worse KL judges it than PCA does')
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('KL vs PCA loss across λ — the PCA metric stays flat (~0.45) while '
                 'KL improves; their ratio collapses from evar', y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'kl_vs_pca_across_lambda')


def fig_perbin_temporal(res, out_dir, mouse='mouse_0', n=3):
    """Per-time-bin temporal posteriors (decoded_samp) for example trials, faceted by
    variant (rows) × trial (cols). Faint = the 10 per-bin posteriors, bold = their
    mean (the trial posterior), grey = target. Reveals whether the shape term
    smooths the PER-BIN distributions or only the Jensen-averaged trial one."""
    names, _ = _lambda_order(list(res.keys()))
    tgt = np.asarray(res['PCA (evar)'][mouse]['Dist']['temp']['target'], float)
    order = np.argsort(np.argmax(tgt, 1))
    picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
    x = np.arange(tgt.shape[1])
    fig, axes = plt.subplots(len(names), n, figsize=(3.4 * n, 1.9 * len(names)),
                             squeeze=False, sharex=True)
    for r, name in enumerate(names):
        ds = np.asarray(res[name][mouse]['Dist']['temp']['decoded_samp'], float)  # (trials,cats,bins)
        for c, tr in enumerate(picks):
            ax = axes[r][c]
            ax.fill_between(x, tgt[tr], color='0.8', alpha=0.7, lw=0)
            nb = ds.shape[2]
            for t in range(nb):
                ax.plot(x, ds[tr, :, t], color=VCOL[name], lw=0.6, alpha=0.35)
            ax.plot(x, ds[tr].mean(1), color=VCOL[name], lw=2.0)  # trial posterior
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'trial {tr}', fontsize=9)
            if c == 0:
                pb = ds[tr].max(0).mean()   # mean per-bin max-prob
                ax.set_ylabel(f'{name}\nper-bin mp={pb:.2f}', fontsize=7.5, color=VCOL[name])
            if r == len(names) - 1:
                ax.set_xlabel('orientation bin', fontsize=8)
    fig.suptitle('Per-time-bin temporal posteriors (faint) and their mean (bold) — '
                 f'temp, {mouse}; grey = target. Does the fix smooth the per-bin dists?', y=1.005)
    fig.tight_layout()
    _save(fig, out_dir, 'perbin_temporal')


def fig_peakiness(peaki, tgt_mp, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    names = list(peaki.keys())
    for ax, (a, lab) in zip(axes, ARCHS):
        for i, name in enumerate(names):
            v = np.asarray(peaki[name][a]['maxprob'])
            ax.bar(i, v.mean(), 0.6, color=VCOL[name], alpha=0.55,
                   yerr=v.std(ddof=1) / np.sqrt(len(v)), capsize=4)
            ax.scatter(np.full(len(v), i) + np.random.uniform(-0.12, 0.12, len(v)),
                       v, color=VCOL[name], s=18, zorder=3, edgecolor='k', lw=0.3)
        ax.axhline(tgt_mp, ls='--', color='k', lw=1.4, label=f'IO target ({tgt_mp:.3f})')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel('decoded max-probability'); ax.set_title(f'{lab} ({a})')
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Decoded peakiness by variant — shape term lands near the target', y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, 'peakiness')


def fig_spat_vs_temp(data, met, out_dir):
    """Two panels (raw, skill): grouped bars by variant, spat vs temp, per-mouse
    dots, paired-t spat-vs-temp p annotated."""
    names = list(data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, ylab in ((axes[0], 'real', f'{met} loss (raw)'),
                          (axes[1], 'skill', f'{met} skill = loss / shuffle')):
        for i, name in enumerate(names):
            for j, (a, alab) in enumerate(ARCHS):
                v = np.asarray(data[name][a][met][key], float)
                xpos = i + (j - 0.5) * 0.36
                col = VCOL[name]
                ax.bar(xpos, np.nanmean(v), 0.34, color=col, alpha=0.4 if j == 0 else 0.8,
                       yerr=np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum()), capsize=3,
                       hatch='' if j == 0 else '//',
                       label=(alab if i == 0 else None))
                ax.scatter(np.full(len(v), xpos), v, color='k', s=10, zorder=3, alpha=0.6)
            # paired-t spat vs temp across mice
            s = np.asarray(data[name]['spat'][met][key], float)
            t = np.asarray(data[name]['temp'][met][key], float)
            good = np.isfinite(s) & np.isfinite(t)
            if good.sum() >= 3:
                p = stats.ttest_rel(s[good], t[good]).pvalue
                ax.text(i, ax.get_ylim()[1] * 0.95, f'p={p:.3f}', ha='center', fontsize=7.5)
        if key == 'skill':
            ax.axhline(1.0, ls=':', color='0.5', lw=1, label='chance')
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel(ylab); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'Spatial vs temporal under {met} — raw & shuffle-normalised, '
                 'across mice (paired-t, n=6)', y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f'spat_vs_temp_{met}')


def fig_within(data, met, out_dir):
    """Per-mouse spat-vs-temp skill, faceted by variant (within-mouse paired)."""
    names = list(data.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(2.5 * len(names), 3.8), sharey=True)
    for ax, name in zip(axes, names):
        s = np.asarray(data[name]['spat'][met]['skill'], float)
        t = np.asarray(data[name]['temp'][met]['skill'], float)
        for k in range(len(s)):
            ax.plot([0, 1], [s[k], t[k]], color=VCOL[name], lw=1.2, alpha=0.7, marker='o', ms=4)
        ax.axhline(1.0, ls=':', color='0.5', lw=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['spatial', 'temporal'])
        ax.set_title(name, color=VCOL[name], fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(f'{met} skill (loss/shuffle); <1 beats chance')
    fig.suptitle(f'Within-mouse spatial vs temporal — {met} skill (one line per mouse)', y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, f'within_mice_{met}')


def write_csv(data, peaki, out_dir):
    rows = ['variant,arch,metric,real_mean,real_sem,shuffle_mean,skill_mean,skill_sem,'
            'maxprob_mean,spat_vs_temp_p']
    for name in data:
        for a, _ in ARCHS:
            for met in METRICS:
                d = data[name][a][met]
                real = np.asarray(d['real'], float); sk = np.asarray(d['skill'], float)
                s = np.asarray(data[name]['spat'][met]['skill'], float)
                t = np.asarray(data[name]['temp'][met]['skill'], float)
                good = np.isfinite(s) & np.isfinite(t)
                p = stats.ttest_rel(s[good], t[good]).pvalue if good.sum() >= 3 else np.nan
                mp = np.mean(peaki[name][a]['maxprob'])
                rows.append(f'{name},{a},{met},{np.nanmean(real):.5f},'
                            f'{np.nanstd(real, ddof=1)/np.sqrt(np.isfinite(real).sum()):.5f},'
                            f'{np.nanmean(d["shuf"]):.5f},{np.nanmean(sk):.5f},'
                            f'{np.nanstd(sk, ddof=1)/np.sqrt(np.isfinite(sk).sum()):.5f},'
                            f'{mp:.4f},{p:.4f}')
    (out_dir / 'summary.csv').write_text('\n'.join(rows) + '\n')
    print(f'  -> summary.csv')


def _save(fig, out_dir, stem):
    ps.save_fig(fig, out_dir, stem)


def main(results_root, split, out_root):
    ps.apply()
    np.random.seed(0)
    global VARIANTS, VCOL
    VARIANTS = discover_variants(results_root)
    VCOL = variant_colors(VARIANTS)
    res, data, peaki, mice, tgt_mp = collect(results_root, split)
    print(f'Variants found: {list(res.keys())}  | mice: {len(mice)}  | target max-prob {tgt_mp:.3f}')
    for name in res:
        for a, _ in ARCHS:
            print(f'  {name:18s} {a}: max-prob {np.mean(peaki[name][a]["maxprob"]):.3f}  '
                  f'PCA-skill {np.nanmean(data[name][a]["PCA"]["skill"]):.3f}  '
                  f'KL-skill {np.nanmean(data[name][a]["KL"]["skill"]):.3f}')
    out_dir = Path(out_root)
    for a, lab in ARCHS:
        fig_examples(res, a, lab, out_dir)
        fig_examples_grid(res, a, lab, out_dir)
    fig_peakiness(peaki, tgt_mp, out_dir)
    fig_metric_vs_lambda(data, out_dir)
    fig_perbin_temporal(res, out_dir)
    for met in METRICS:
        fig_spat_vs_temp(data, met, out_dir)
        fig_within(data, met, out_dir)
    write_csv(data, peaki, out_dir)
    print(f'Done. {out_dir.resolve()}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split', default='stratified_balanced')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/loss_variants')
    a = ap.parse_args()
    main(a.results_root, a.split, a.out_root)
