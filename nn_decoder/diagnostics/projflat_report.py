# -*- coding: utf-8 -*-
"""Report for `projflat_v1` — projection loss + FLAT weighting (= MSE over 91 bins),
architecture x input dimensionality x regularisation, at patience 20.

Three axes, per the standing rule, every cell judged on all three:
  peakiness   — decoded max-prob / IO target        (bias; 1.0 = on target)
  overfitting — val / train fit-loss ratio           (variance; 1.0 = none)
  performance — held-out loss / LOO predict-mean, under BOTH KL and the
                projection metric on a COMMON evar weighting  (<1 beats chance)

An annihilated decoder scores 1.00 on overfitting and below target on peakiness,
so it can only be caught on the performance axis — hence all three, always.
Collapsed / suppressed cells are marked, never silently plotted as a win.

Figures (PNG+SVG, figures/projflat/):
  fig1_dim_weighting  — input dimensionality (raw/3/5/10 PCs) x weighting (flat vs
                        evar), both architectures, all four measures. The base+evar
                        arms; answers "does dim reduction help, and does MSE beat
                        the eigenvalue weighting" at patience 20.
  fig2_reg_spatial    — the dropout x wd regularisation grid, spatial decoder
                        (lambda_H is temporal-only, so spatial is read at lambda_H=0).
  fig3_reg_temporal   — the lambda_H x dropout x wd grid, temporal decoder.

Also writes metrics.csv (one row per cell x arch, all measures + the guard flags).

Usage:
  python diagnostics/projflat_report.py
  python diagnostics/projflat_report.py --only dim
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_vs_hparams import _norm_by_mouse  # noqa: E402
from cross_loss_eval import _eval_one  # noqa: E402
from overfitting_vs_hparams import _overfit_ratio  # noqa: E402

RUN = 'projflat_v1'
SPLIT = 'stratified_balanced'
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]
INPUTS = ['raw', 'pc3', 'pc5', 'pc10']
LAMBDAS = [0.0, 3e-3, 1e-2]
DROPOUTS = [0.0, 0.25, 0.5]
WDECAYS = [0.0, 1e-6, 1e-5]

_CACHE: dict = {}


# ------------------------------------------------------------------- loading
def _slug(results_root, cell):
    d = Path(results_root) / RUN / cell
    if not d.is_dir():
        return None
    subs = [p for p in d.iterdir() if p.is_dir() and (p / f'{SPLIT}.mat').is_file()]
    return subs[0] if subs else None


def have(results_root, cell):
    return _slug(results_root, cell) is not None


def _res(results_root, cell):
    key = (str(results_root), cell)
    if key not in _CACHE:
        d = _slug(results_root, cell)
        if d is None:
            raise KeyError(cell)
        _CACHE[key] = sio.loadmat(str(d / f'{SPLIT}.mat'),
                                  simplify_cells=True)['results']
    return _CACHE[key]


def _mice(r):
    return sorted(k for k in r if isinstance(r[k], dict)
                  and isinstance(r[k].get('Dist'), dict))


# ------------------------------------------------- the common projection basis
def _common_basis(results_root, arch, anchor='h8_raw_EVAR'):
    """(pcs, evar) per mouse from an evar-weighted cell — the shared projection
    metric, so flat and evar decoders are scored under identical weights."""
    r = _res(results_root, anchor)
    return {m: (r[m]['Dist'].get('pcs'), r[m]['Dist'].get('explained_var'))
            for m in _mice(r)}


def _proj_common(r, arch, basis):
    vals = []
    for m in _mice(r):
        if m not in basis:
            continue
        pcs, evar = basis[m]
        D = r[m]['Dist']
        dec = np.asarray(D[arch]['decoded'], float)
        tgt = np.asarray(D[arch]['target'], float)
        ok = np.isfinite(tgt).all(1)
        if not ok.any():
            continue
        n_ok = int(ok.sum())
        tot = tgt[ok].sum(axis=0)
        pm = np.tile((tot / n_ok)[None, :], (tgt.shape[0], 1))
        if n_ok > 1:
            pm[ok] = (tot[None, :] - tgt[ok]) / (n_ok - 1)
        num = _eval_one(dec, tgt, 'PCA', pcs, evar)
        den = _eval_one(pm, tgt, 'PCA', pcs, evar)
        if np.isfinite(num) and np.isfinite(den) and den > 0:
            vals.append(num / den)
    return np.array(vals, float)


# --------------------------------------------------------------- the measures
def measures(results_root, cell, arch, basis):
    """Per-mouse arrays: (peaky/target, overfit ratio, KL normloss, proj normloss)."""
    r = _res(results_root, cell)
    ms = _mice(r)
    pk = np.array([np.asarray(r[m]['Dist'][arch]['decoded'], float).max(1).mean() for m in ms])
    tg = np.array([np.asarray(r[m]['Dist'][arch]['target'], float).max(1).mean() for m in ms])
    kl = np.array(_norm_by_mouse(r, arch)[('KL', 'pm')], float)
    pj = _proj_common(r, arch, basis)
    slug = _slug(results_root, cell)
    of, _ = _overfit_ratio(slug / 'checkpoints', arch)
    return dict(ratio=pk / tg, overfit=of, kl=kl, proj=pj,
                peaky=pk, target=tg)


def weight_norm(results_root, cell, arch):
    r = _res(results_root, cell)
    return np.array([np.linalg.norm(np.asarray(r[m]['Weights'][arch]['W_in'], float))
                     for m in _mice(r)])


def suspect(results_root, cell, arch, basis):
    """Collapsed (|W|~0 and peaky~1/n_cats) OR suppressed (broader than target and
    worse than chance under KL)."""
    try:
        m = measures(results_root, cell, arch, basis)
        wn = weight_norm(results_root, cell, arch).mean()
    except Exception:                                    # noqa: BLE001
        return False
    r = _res(results_root, cell)
    n_cats = np.asarray(r[_mice(r)[0]]['Dist'][arch]['decoded'], float).shape[1]
    collapsed = wn < 1e-3 and abs(m['peaky'].mean() - 1.0 / n_cats) < 1e-3
    suppressed = m['ratio'].mean() < 1.0 and m['kl'].mean() >= 1.0
    return bool(collapsed or suppressed)


def _msem(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


def _require(results_root, cells, what):
    miss = [c for c in cells if not have(results_root, c)]
    if miss:
        print(f"  [skip] {what}: {len(miss)}/{len(cells)} cells absent -> "
              f"{', '.join(miss[:6])}{'...' if len(miss) > 6 else ''}")
        return False
    return True


MEAS_ROWS = [('ratio', 'peakiness / IO target', 1.0, False),
             ('overfit', 'val / train fit-loss', 1.0, False),
             ('kl', 'norm. loss under KL', 1.0, True),
             ('proj', 'norm. loss under PROJECTION', 1.0, True)]


# -------------------------------------------------- fig 1: dimensionality x weighting
def fig_dim_weighting(results_root, out_root):
    # flat base cells: <arch>_<input>_l0_d0_w0 ; evar: <arch>_<input>_EVAR
    need = []
    for arch_tok in ('h8', 'lin'):
        for i in INPUTS:
            need += [f'{arch_tok}_{i}_l0_d0_w0', f'{arch_tok}_{i}_EVAR']
        need.append(f'{arch_tok}_raw_KLref')
    if not _require(results_root, need, 'fig1 dim x weighting'):
        return
    ps.apply()
    basis = {a: _common_basis(results_root, a) for a, _ in ARCHS}
    fig, axes = plt.subplots(len(MEAS_ROWS), 2, figsize=ps.figsize(2, 4), sharex=True)
    xtok = {'raw': 0, 'pc3': 1, 'pc5': 2, 'pc10': 3}
    for ci, (arch, alab) in enumerate(ARCHS):
        b = basis[arch]
        for arch_tok, hs_lab, ls in [('h8', 'H=8', '-'), ('lin', 'linear', '--')]:
            flat_m = {i: measures(results_root, f'{arch_tok}_{i}_l0_d0_w0', arch, b) for i in INPUTS}
            evar_m = {i: measures(results_root, f'{arch_tok}_{i}_EVAR', arch, b) for i in INPUTS}
            for ri, (key, ylab, ref, _islog) in enumerate(MEAS_ROWS):
                ax = axes[ri][ci]
                xs = [xtok[i] for i in INPUTS]
                for mdict, colr, wlab in [(flat_m, ps.FLAT_EVAR, 'flat/MSE'),
                                          (evar_m, ps.PCA_EVAR, 'evar')]:
                    ys = [_msem(mdict[i][key])[0] for i in INPUTS]
                    es = [_msem(mdict[i][key])[1] for i in INPUTS]
                    ax.errorbar(xs, ys, yerr=es, marker='o', ms=4, lw=1.4, ls=ls,
                                color=colr, capsize=3,
                                label=f'{wlab}, {hs_lab}' if ri == 0 and ci == 0 else None)
        # KL reference line (raw) on the performance rows
        for ri, (key, ylab, ref, islog) in enumerate(MEAS_ROWS):
            ax = axes[ri][ci]
            if key in ('kl', 'proj'):
                klm = measures(results_root, 'h8_raw_KLref', arch, b)
                ax.axhline(_msem(klm[key])[0], color=ps.KL, ls=':', lw=1.2,
                           label='KL ref (H=8)' if ri == 2 and ci == 0 else None)
            ax.axhline(ref, color='k', ls=':', lw=1.0, alpha=0.6)
            if islog or key == 'overfit':
                ax.set_yscale('log')
            ax.set_xticks(list(xtok.values()))
            ax.set_xticklabels(list(xtok), fontsize=7)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=7.5)
            if ri == 0:
                ax.set_title(alab, fontsize=11, fontweight='bold')
            if ri == len(MEAS_ROWS) - 1:
                ax.set_xlabel('neural input (raw or #PCs)', fontsize=8)
    axes[0][0].legend(fontsize=6, frameon=True, loc='best', ncol=2)
    ps.label_panels(axes.ravel())
    fig.suptitle('projflat_v1 — input dimensionality x weighting, patience 20, un-regularised. '
                 'flat/MSE (blue) vs eigenvalue weighting (orange); solid H=8, dashed linear. '
                 'Dotted black = on target / chance; dotted purple = KL reference.',
                 y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'projflat_fig1_dim_weighting')


def _tok(v):
    if v == 0:
        return '0'
    if v < 1e-3:
        return f'{v:.0e}'.replace('e-0', 'em').replace('e-', 'em')
    return str(v).replace('.', 'p')


def _grid_cell(arch_tok, lam, drop, wd):
    return f'{arch_tok}_raw_l{_tok(lam)}_d{_tok(drop)}_w{_tok(wd)}'


# ------------------------------------------- fig 2/3: the regularisation grid
def _fig_reg(results_root, out_root, arch, alab, arch_tok, lam_levels, stem, title):
    cells = [_grid_cell(arch_tok, lam, d, w)
             for lam in lam_levels for d in DROPOUTS for w in WDECAYS]
    if not _require(results_root, cells, stem):
        return
    ps.apply()
    b = _common_basis(results_root, arch)
    ncol = len(lam_levels)
    fig, axes = plt.subplots(len(MEAS_ROWS), ncol,
                             figsize=ps.figsize(ncol, len(MEAS_ROWS)),
                             squeeze=False, sharex=True)
    dcolors = ps.SHAPE_GREENS[1:4] if hasattr(ps, 'SHAPE_GREENS') else ['#74c476', '#41ab5d', '#005a32']
    xw = list(range(len(WDECAYS)))
    klm = measures(results_root, f'{arch_tok}_raw_KLref', arch, b) \
        if have(results_root, f'{arch_tok}_raw_KLref') else None
    for cj, lam in enumerate(lam_levels):
        for ri, (key, ylab, ref, islog) in enumerate(MEAS_ROWS):
            ax = axes[ri][cj]
            for di, drop in enumerate(DROPOUTS):
                ys, es, dead = [], [], []
                for wd in WDECAYS:
                    cell = _grid_cell(arch_tok, lam, drop, wd)
                    m = measures(results_root, cell, arch, b)
                    mm, ss = _msem(m[key])
                    ys.append(mm)
                    es.append(ss)
                    dead.append(suspect(results_root, cell, arch, b))
                ax.errorbar(xw, ys, yerr=es, marker='o', ms=4, lw=1.3,
                            color=dcolors[di], capsize=3,
                            label=f'dropout {drop}' if (ri == 0 and cj == 0) else None)
                if any(dead):
                    ax.plot([i for i, x in enumerate(dead) if x],
                            [y for y, x in zip(ys, dead) if x], 'x', ms=8, mew=2,
                            color='k', ls='none', zorder=6)
            if key in ('kl', 'proj') and klm is not None:
                ax.axhline(_msem(klm[key])[0], color=ps.KL, ls=':', lw=1.1)
            ax.axhline(ref, color='k', ls=':', lw=1.0, alpha=0.6)
            if islog or key == 'overfit':
                ax.set_yscale('log')
            ax.set_xticks(xw)
            ax.set_xticklabels(['0', '1e-6', '1e-5'], fontsize=6.5)
            if ri == 0:
                ax.set_title(f'λ_H = {lam}', fontsize=9)
            if cj == 0:
                ax.set_ylabel(ylab, fontsize=7)
            if ri == len(MEAS_ROWS) - 1:
                ax.set_xlabel('weight decay', fontsize=7.5)
    axes[0][0].legend(fontsize=6, frameon=True, loc='best')
    ps.label_panels(axes.ravel())
    fig.suptitle(title, y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), stem)


def fig_reg_spatial(results_root, out_root):
    # spatial is lambda_H-invariant (entropy penalty is temporal-only); show lam=0 only
    _fig_reg(results_root, out_root, 'spat', 'spatial', 'h8', [0.0],
             'projflat_fig2_reg_spatial',
             'Regularisation grid — SPATIAL (H=8, flat/MSE, patience 20). λ_H is '
             'temporal-only so spatial is read at λ_H=0. x = weight decay, colour = '
             'dropout. Dotted purple = KL ref; X = collapsed.')


def fig_reg_temporal(results_root, out_root):
    _fig_reg(results_root, out_root, 'temp', 'temporal', 'h8', LAMBDAS,
             'projflat_fig3_reg_temporal',
             'Regularisation grid — TEMPORAL (H=8, flat/MSE, patience 20). Columns = '
             'λ_H (the SBC entropy penalty). x = weight decay, colour = dropout. '
             'Dotted purple = KL ref; X = collapsed toward uniform.')


# ------------------------------------------------------- metrics.csv + headline
def write_metrics(results_root, out_root):
    root = Path(results_root) / RUN
    if not root.is_dir():
        print("  [skip] metrics.csv: run not present")
        return []
    cells = sorted(p.name for p in root.iterdir() if p.is_dir() and have(results_root, p.name))
    basis = {a: _common_basis(results_root, a) for a, _ in ARCHS}
    rows = []
    for cell in cells:
        for arch, _ in ARCHS:
            try:
                m = measures(results_root, cell, arch, basis[arch])
            except Exception as e:                       # noqa: BLE001
                print(f"  [warn] {cell}/{arch}: {e}")
                continue
            rows.append(dict(
                cell=cell, arch=arch, n_mice=m['ratio'].size,
                peaky=round(m['peaky'].mean(), 4),
                io_target=round(m['target'].mean(), 4),
                over_sharpening=round(m['ratio'].mean(), 3),
                overfit=round(float(m['overfit']), 3) if m['overfit'] else '',
                kl_loss=round(m['kl'].mean(), 4),
                proj_loss=round(m['proj'].mean(), 4) if m['proj'].size else '',
                kl_beats_chance=int((m['kl'] < 1).sum()),
                weight_norm=round(float(weight_norm(results_root, cell, arch).mean()), 4),
                suspect=int(suspect(results_root, cell, arch, basis[arch])),
            ))
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out / 'metrics.csv'}  ({len(rows)} rows, {len(cells)} cells)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    ap.add_argument('--only', nargs='+', default=None,
                    choices=['dim', 'regspat', 'regtemp', 'metrics'])
    a = ap.parse_args()
    want = a.only or ['dim', 'regspat', 'regtemp', 'metrics']
    print(f"projflat_v1 report -> {a.out_root}")
    if 'dim' in want:
        fig_dim_weighting(a.results_root, a.out_root)
    if 'regspat' in want:
        fig_reg_spatial(a.results_root, a.out_root)
    if 'regtemp' in want:
        fig_reg_temporal(a.results_root, a.out_root)
    if 'metrics' in want:
        write_metrics(a.results_root, a.out_root)


if __name__ == '__main__':
    main()
