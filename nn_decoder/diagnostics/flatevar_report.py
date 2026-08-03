# -*- coding: utf-8 -*-
"""Report for the `flatevar_v1` run (Q/half/100 ms, 6 mice, restart selection on
validation, seed 0) — the 2026-07-29 meeting asks.

The run takes the width fix to its limit: `flat_evar=True` is the `shape_lambda -> inf`
case (uniform per-PC weights, i.e. the projection loss becomes an unweighted L2/Brier
distance). Three questions, three figures.

  fig1  CORE — does flat weighting fix the over-sharpening, with and without a hidden
        layer? `B_flat_linear` (flat weighting + ZERO hidden units) is the decisive
        cell: prodfix_v1 arm C showed the projection loss over-sharpens 5.6x/10.5x with
        no hidden layer, killing the capacity account. If the evar weighting is the
        cause, removing it must fix it even with no capacity present.
  fig2  NEURAL-PC LADDER — decode from the leading k principal components of the neural
        activity (input side, basis fit on training trials only). Does a low-dimensional
        subspace suffice, and does input-side PCA alone change anything (the
        `C_evar_npc16` control)?
  fig3  SWEEP — how much do the other hyperparameters matter once the weighting is flat?
        One-at-a-time over width / dropout / weight decay / lambda_H / early stopping,
        for H=8 and for the linear decoder.

EVERY CELL IS JUDGED TWICE — decoded peakiness against the IO target AND normalised
loss (held-out KL(decoded || target) / leave-one-out predict-mean; < 1 beats chance).
Landing peakiness without beating chance is a lobotomy, not a cure: `weight_decay=0.01`
drives peakiness to 0.011 = 1/91 = the uniform decoder while its normalised loss sits
at ~1.6. Peakiness alone would have called that a fix.

MISSING CELLS: each figure REFUSES to draw unless every cell it needs is present, and
says which are absent. A figure drawn from half a run, with a title still asserting the
conclusion, is worse than no figure (2026-07-16 audit).

Outputs (PNG+SVG) + metrics.csv under figures/flatevar/.
Usage:
    python diagnostics/flatevar_report.py
    python diagnostics/flatevar_report.py --only core
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import peakiness_style as ps  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_vs_hparams import _norm_by_mouse  # noqa: E402

RUN = 'flatevar_v1'
SPLIT = 'stratified_balanced'
ARCHS = [('spat', 'spatial'), ('temp', 'temporal')]

_CACHE: dict = {}


# --------------------------------------------------------------------- loading
def _cell_dir(results_root, cell):
    d = Path(results_root) / RUN / cell
    if not d.is_dir():
        return None
    subs = [p for p in d.iterdir() if p.is_dir()]
    if not subs:
        return None
    slug = subs[0]                     # discovered, not assumed
    return slug if (slug / f'{SPLIT}.mat').is_file() else None


def available(results_root, cell):
    return _cell_dir(results_root, cell) is not None


def _res(results_root, cell):
    key = (str(results_root), cell)
    if key not in _CACHE:
        d = _cell_dir(results_root, cell)
        if d is None:
            raise KeyError(cell)
        _CACHE[key] = sio.loadmat(str(d / f'{SPLIT}.mat'),
                                  simplify_cells=True)['results']
    return _CACHE[key]


def _mice(res):
    return sorted(k for k in res
                  if isinstance(res[k], dict) and isinstance(res[k].get('Dist'), dict))


def peaky(results_root, cell, arch):
    """Per-mouse decoded peakiness = mean over trials of max-probability."""
    r = _res(results_root, cell)
    return np.array([np.asarray(r[m]['Dist'][arch]['decoded'], float).max(1).mean()
                     for m in _mice(r)])


def target_peak(results_root, cell, arch):
    r = _res(results_root, cell)
    return np.array([np.asarray(r[m]['Dist'][arch]['target'], float).max(1).mean()
                     for m in _mice(r)])


def normloss(results_root, cell, arch):
    """Held-out KL(decoded || target) / leave-one-out predict-mean. <1 beats chance."""
    return np.array(_norm_by_mouse(_res(results_root, cell), arch)[('KL', 'pm')], float)


def _msem(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)


def weight_norm(results_root, cell, arch):
    """Per-mouse Frobenius norm of the input weights. Zero = annihilated network."""
    r = _res(results_root, cell)
    return np.array([np.linalg.norm(np.asarray(r[m]['Weights'][arch]['W_in'], float))
                     for m in _mice(r)])


def collapsed(results_root, cell, arch, tol=1e-3):
    """Has this decoder collapsed to the uniform output?

    Two independent signatures, both required to be safe: the input weights are
    numerically zero, AND the decoded peakiness sits on 1/n_cats. Under flat
    per-PC weighting every weight is 1/n_pc, so the fit-loss gradient is ~45x
    weaker than under the evar weighting, while Adam's weight-decay term is
    unchanged — the decay wins and drives theta to exactly 0. Such a cell carries
    NO information about the knob being swept, and must never be plotted as if
    it did. (Found 2026-08-03: at the run's baseline wd=1e-4 this killed most of
    the flat_evar arm.)
    """
    try:
        wn = weight_norm(results_root, cell, arch)
        pk = peaky(results_root, cell, arch)
    except Exception:                                   # noqa: BLE001
        return False
    r = _res(results_root, cell)
    n_cats = np.asarray(r[_mice(r)[0]]['Dist'][arch]['decoded'], float).shape[1]
    return bool((wn.mean() < tol) and (abs(pk.mean() - 1.0 / n_cats) < tol))


def lobotomised(results_root, cell, arch):
    """Under-fit toward uniform: BROADER than the IO target yet WORSE than chance.

    The hard `collapsed` test only catches weights driven to exactly zero. Weight
    decay also produces partial suppression — `B_flat_linear` spatial has
    |W_in| = 0.42 against 23.0 for the same cell at wd=0, i.e. suppressed 55x
    without hitting zero. Such a decoder is still uninformative about the knob
    being swept, and this quadrant (peakiness < target AND normalised loss >= 1)
    identifies it without needing a matched wd=0 twin to compare against.

    Note this is a DIFFERENT failure from "sharper than target and worse than
    chance" (B_flat_lin_wd0 spatial), which is a genuinely bad decoder rather
    than a suppressed one — that case is deliberately not flagged.
    """
    try:
        return bool(peaky(results_root, cell, arch).mean()
                    < target_peak(results_root, cell, arch).mean()
                    and normloss(results_root, cell, arch).mean() >= 1.0)
    except Exception:                                   # noqa: BLE001
        return False


def suspect(results_root, cell, arch):
    return (collapsed(results_root, cell, arch)
            or lobotomised(results_root, cell, arch))


def _require(results_root, cells, what):
    """Return True if every cell is present; otherwise say what is missing and skip."""
    missing = [c for c in cells if not available(results_root, c)]
    if missing:
        print(f"  [skip] {what}: {len(missing)}/{len(cells)} cells not downloaded yet "
              f"-> {', '.join(missing)}")
        return False
    return True


# ------------------------------------------------------------------ fig 1: core
CORE = [
    ('R_evar_base',    'evar\nH=8',      ps.PCA_EVAR, ''),
    ('A_flat_base',    'FLAT\nH=8',      ps.FLAT_EVAR, ''),
    ('R_evar_linear',  'evar\nlinear',   ps.PCA_EVAR, '//'),
    ('B_flat_linear',  'FLAT\nlinear',   ps.FLAT_EVAR, '//'),
    ('R_reference_kl', 'KL',             ps.KL, ''),
    ('R_reference_js', 'JS',             ps.JS, ''),
]


def fig_core(results_root, out_root):
    cells = [c[0] for c in CORE]
    if not _require(results_root, cells, 'fig1 core'):
        return
    ps.apply()
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2))
    for col, (arch, alab) in enumerate(ARCHS):
        tgt = target_peak(results_root, 'A_flat_base', arch).mean()
        for row, metric in enumerate(('peak', 'norm')):
            ax = axes[row][col]
            for j, (cell, lab, colr, hatch) in enumerate(CORE):
                v = (peaky(results_root, cell, arch) if metric == 'peak'
                     else normloss(results_root, cell, arch))
                m, s = _msem(v)
                bad = suspect(results_root, cell, arch)
                ax.bar(j, m, 0.7, yerr=s, capsize=3, color=colr, hatch=hatch,
                       edgecolor='k', linewidth=0.5, alpha=0.35 if bad else 1.0)
                # A suppressed decoder sits BELOW the IO target on the peakiness
                # panel, which reads as a triumph unless it is called out. Mark it
                # on the bar itself, not only via the loss panel below.
                if bad:
                    ax.plot(j, m, 'x', ms=11, mew=2.5, color='k', zorder=5)
            ax.set_xticks(range(len(CORE)))
            ax.set_xticklabels([c[1] for c in CORE], fontsize=7)
            if metric == 'peak':
                ax.axhline(tgt, color=ps.TARGET_LINE, ls=':', lw=1.4)
                ax.set_title(alab, fontsize=10)
                if col == 0:
                    ax.set_ylabel('decoded peakiness\n(dotted = IO target)')
            else:
                ax.axhline(1.0, color=ps.TARGET_LINE, ls=':', lw=1.4)
                ax.set_yscale('log')
                if col == 0:
                    ax.set_ylabel('normalised loss / predict-mean\n(dotted = chance)')
    axes[0][0].legend(handles=[Line2D([0], [0], color='k', ls=':', lw=1.4),
                               Line2D([0], [0], color='k', marker='x', ls='none',
                                      ms=9, mew=2.5)],
                      labels=['IO target', 'suppressed toward uniform\n(faded; not a cure)'],
                      fontsize=6.5, frameon=True)
    ps.label_panels(axes.ravel())
    fig.suptitle('Flat per-PC weighting (shape_lambda -> inf) vs the evar weighting, with and without a '
                 'hidden layer. At the run baseline wd=1e-4 the FLAT cells are SUPPRESSED toward the uniform '
                 'decoder (x, faded) — low peakiness (a, b) with worse-than-chance loss (c, d) is a lobotomy, '
                 'not a cure. See fig3c for the clean wd=0 comparison. (6 mice, mean±sem)',
                 y=1.03, fontsize=8)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'flatevar_fig1_core')


# ---------------------------------------------------------- fig 2: neural PCs
LADDER_H8 = [('C_npc2', 2), ('C_npc4', 4), ('C_npc8', 8),
             ('C_npc16', 16), ('C_npc32', 32), ('C_npc64', 64)]
LADDER_LIN = [('C_lin_npc4', 4), ('C_lin_npc16', 16), ('C_lin_npc64', 64)]


def fig_neural_pcs(results_root, out_root):
    need = [c for c, _ in LADDER_H8 + LADDER_LIN] + \
           ['A_flat_base', 'B_flat_linear', 'C_evar_npc16']
    if not _require(results_root, need, 'fig2 neural-PC ladder'):
        return
    ps.apply()
    fig, axes = plt.subplots(2, 2, figsize=ps.figsize(2, 2))
    for col, (arch, alab) in enumerate(ARCHS):
        tgt = target_peak(results_root, 'A_flat_base', arch).mean()
        for row, metric in enumerate(('peak', 'norm')):
            ax = axes[row][col]
            fn = peaky if metric == 'peak' else normloss
            for ladder, full_cell, lab, colr, ls in [
                    (LADDER_H8, 'A_flat_base', 'flat, H=8', ps.FLAT_EVAR, '-'),
                    (LADDER_LIN, 'B_flat_linear', 'flat, linear', ps.SHAPE, '--')]:
                ks = [k for _, k in ladder]
                ms, ss = zip(*[_msem(fn(results_root, c, arch)) for c, _ in ladder])
                ax.errorbar(ks, ms, yerr=ss, marker='o', ms=4, lw=1.4, ls=ls,
                            color=colr, capsize=3, label=lab)
                fm, _ = _msem(fn(results_root, full_cell, arch))
                ax.axhline(fm, color=colr, ls=':', lw=1.0, alpha=0.8)
            em, es = _msem(fn(results_root, 'C_evar_npc16', arch))
            ax.errorbar([16], [em], yerr=[es], marker='s', ms=6, color=ps.PCA_EVAR,
                        capsize=3, ls='none', label='evar weighting, k=16 (control)')
            ax.set_xscale('log', base=2)
            ax.set_xticks([k for _, k in LADDER_H8])
            ax.set_xticklabels([str(k) for _, k in LADDER_H8], fontsize=7)
            ax.set_xlabel('neural PCs retained (k)')
            if metric == 'peak':
                ax.axhline(tgt, color=ps.TARGET_LINE, ls=':', lw=1.4)
                ax.set_title(alab, fontsize=10)
                if col == 0:
                    ax.set_ylabel('decoded peakiness\n(dotted black = IO target)')
                    ax.legend(fontsize=6.5, frameon=True)
            else:
                ax.axhline(1.0, color=ps.TARGET_LINE, ls=':', lw=1.4)
                ax.set_yscale('log')
                if col == 0:
                    ax.set_ylabel('normalised loss / predict-mean\n(dotted = chance)')
    ps.label_panels(axes.ravel())
    fig.suptitle('Decoding from the leading k neural PCs (basis fit on training trials only). '
                 'Coloured dotted lines = the same decoder on the FULL population.',
                 y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), 'flatevar_fig2_neural_pcs')


# ------------------------------------------------------------- fig 3: OAT sweep
# (axis label, [(cell, tick label)]) — the baseline cell is spliced in at its value.
SWEEP_H8 = [
    ('width (H)',   [('A_flat_h2', '2'), ('A_flat_h4', '4'), ('A_flat_base', '8*'),
                     ('A_flat_h16', '16'), ('A_flat_h32', '32')]),
    ('dropout',     [('A_flat_base', '0*'), ('A_flat_drop0p25', '.25'),
                     ('A_flat_drop0p5', '.5')]),
    ('weight decay', [('A_flat_wd0', '0'), ('A_flat_base', '1e-4*'),
                      ('A_flat_wd1em3', '1e-3'), ('A_flat_wd1em2', '1e-2')]),
    ('lambda_H',    [('A_flat_lam0', '0'), ('A_flat_base', '3e-3*'),
                     ('A_flat_lam1em2', '1e-2')]),
    ('early stop',  [('A_flat_base', 'off*'), ('A_flat_earlystop', 'pat 20')]),
]
SWEEP_LIN = [
    ('dropout',     [('B_flat_linear', '0*'), ('B_flat_lin_drop0p25', '.25'),
                     ('B_flat_lin_drop0p5', '.5')]),
    ('weight decay', [('B_flat_lin_wd0', '0'), ('B_flat_linear', '1e-4*'),
                      ('B_flat_lin_wd1em3', '1e-3'), ('B_flat_lin_wd1em2', '1e-2')]),
    ('lambda_H',    [('B_flat_lin_lam0', '0'), ('B_flat_linear', '3e-3*'),
                     ('B_flat_lin_lam1em2', '1e-2')]),
    ('early stop',  [('B_flat_linear', 'off*'), ('B_flat_lin_earlystop', 'pat 20')]),
]


def _fig_sweep(results_root, out_root, spec, base_cell, stem, title):
    need = sorted({c for _, pts in spec for c, _ in pts})
    if not _require(results_root, need, stem):
        return
    ps.apply()
    ncol = len(spec)
    fig, axes = plt.subplots(2, ncol, figsize=ps.figsize(ncol, 2), squeeze=False)
    for j, (axis_lab, pts) in enumerate(spec):
        for row, metric in enumerate(('peak', 'norm')):
            ax = axes[row][j]
            fn = peaky if metric == 'peak' else normloss
            for arch, alab in ARCHS:
                ms, ss = zip(*[_msem(fn(results_root, c, arch)) for c, _ in pts])
                dead = [suspect(results_root, c, arch) for c, _ in pts]
                ax.errorbar(range(len(pts)), ms, yerr=ss, marker='o', ms=4, lw=1.4,
                            color=ps.ARCH[arch], capsize=3, alpha=0.9,
                            label=alab if j == 0 else None)
                # Mark annihilated cells. Their value is an artefact of weight decay
                # beating a 45x-weaker fit gradient, not a property of the swept knob,
                # so the reader must not read a flat line through them as "this knob
                # does not matter".
                if any(dead):
                    ax.plot([i for i, d in enumerate(dead) if d],
                            [m for m, d in zip(ms, dead) if d], 'x', ms=9, mew=2,
                            color='k', ls='none', zorder=5,
                            label='suppressed toward uniform\n(uninformative about this knob)'
                                  if (j == 0 and arch == 'spat') else None)
            ax.set_xticks(range(len(pts)))
            ax.set_xticklabels([t for _, t in pts], fontsize=7)
            ax.set_xlabel(axis_lab, fontsize=8)
            if metric == 'peak':
                ax.axhline(target_peak(results_root, base_cell, 'spat').mean(),
                           color=ps.TARGET_LINE, ls=':', lw=1.2)
                if j == 0:
                    ax.set_ylabel('peakiness\n(dotted = IO target)', fontsize=8)
                    ax.legend(fontsize=6.5, frameon=True)
            else:
                ax.axhline(1.0, color=ps.TARGET_LINE, ls=':', lw=1.2)
                ax.set_yscale('log')
                if j == 0:
                    ax.set_ylabel('normalised loss\n(dotted = chance)', fontsize=8)
    ps.label_panels(axes.ravel())
    n_dead = sum(suspect(results_root, c, a) for _, pts in spec for c, _ in pts
                 for a, _ in ARCHS)
    warn = ('   ** X = decoder SUPPRESSED toward uniform by weight decay (broader than '
            'target AND worse than chance) — those points say nothing about the swept knob **'
            if n_dead else '')
    fig.suptitle(title + '   (* = the baseline setting; 6 mice, mean±sem)' + warn,
                 y=1.02, fontsize=9)
    fig.tight_layout()
    ps.save_fig(fig, Path(out_root), stem)


def fig_sweep_h8(results_root, out_root):
    _fig_sweep(results_root, out_root, SWEEP_H8, 'A_flat_base',
               'flatevar_fig3_sweep_h8',
               'How much do the other hyperparameters matter once the per-PC weighting '
               'is FLAT? (H=8)')


def fig_sweep_linear(results_root, out_root):
    _fig_sweep(results_root, out_root, SWEEP_LIN, 'B_flat_linear',
               'flatevar_fig4_sweep_linear',
               'The same sweep with ZERO hidden units (flat weighting, linear decoder)')


# ------------------------------------------------------------------- metrics
def write_metrics(results_root, out_root):
    """One row per (cell, arch): peakiness, IO target, over-sharpening ratio,
    normalised loss, and the per-mouse sign consistency against the IO target."""
    root = Path(results_root) / RUN
    cells = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    cells = [c for c in cells if available(results_root, c)]
    if not cells:
        print("  [skip] metrics.csv: no cells downloaded yet")
        return []
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in cells:
        for arch, _ in ARCHS:
            try:
                pk = peaky(results_root, cell, arch)
                tg = target_peak(results_root, cell, arch)
                nl = normloss(results_root, cell, arch)
            except Exception as e:                      # noqa: BLE001
                print(f"  [warn] {cell}/{arch}: {e}")
                continue
            ratio = pk / tg
            pm, psem = _msem(pk)
            nm, nsem = _msem(nl)
            # Paired t vs the IO target: is this decoder's peakiness distinguishable
            # from its own target? n=6 mice. Sign consistency is the robust statement.
            t, p = ttest_rel(pk, tg) if pk.size > 1 else (np.nan, np.nan)
            rows.append(dict(
                cell=cell, arch=arch, n_mice=pk.size,
                peakiness=round(pm, 5), peakiness_sem=round(psem, 5),
                io_target=round(tg.mean(), 5),
                over_sharpening=round(ratio.mean(), 3),
                norm_loss=round(nm, 4), norm_loss_sem=round(nsem, 4),
                collapsed=int(collapsed(results_root, cell, arch)),
                lobotomised=int(lobotomised(results_root, cell, arch)),
                weight_norm=round(float(weight_norm(results_root, cell, arch).mean()), 5),
                beats_chance_mice=int((nl < 1).sum()),
                over_target_mice=int((ratio > 1.5).sum()),
                t_vs_target=round(float(t), 3) if np.isfinite(t) else '',
                p_vs_target=round(float(p), 5) if np.isfinite(p) else '',
            ))
    path = out / 'metrics.csv'
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows, {len(cells)} cells)")
    return rows


def print_headline(rows):
    """The two registered priors, answered — or explicitly reported as not yet answerable."""
    by = {(r['cell'], r['arch']): r for r in rows}
    print("\n" + "=" * 78)
    print("REGISTERED PRIORS (PREDICTIONS.md, 2026-07-29)")
    print("=" * 78)
    for tag, cell, note in [
            ('P1', 'A_flat_base',
             'flat_evar at H=8 lands on target and beats chance (pred ~0.05-0.07, ~0.72-0.85)'),
            ('P2', 'B_flat_linear',
             'DECISIVE: flat + ZERO hidden units also lands on target (pred <=1.3x, >=5/6 mice)')]:
        print(f"\n{tag}  {cell}\n    {note}")
        for arch, alab in ARCHS:
            r = by.get((cell, arch))
            if r is None:
                print(f"    {alab:9s} NOT DOWNLOADED YET")
                continue
            dead = ('  ** ANNIHILATED (|W|=0) — uninformative **' if r['collapsed']
                    else '  ** SUPPRESSED toward uniform — uninformative **'
                    if r['lobotomised'] else '')
            print(f"    {alab:9s} peakiness {r['peakiness']:.4f} vs target "
                  f"{r['io_target']:.4f}  = {r['over_sharpening']:.2f}x   |   "
                  f"normalised loss {r['norm_loss']:.3f} "
                  f"({r['beats_chance_mice']}/{r['n_mice']} mice beat chance){dead}")
    print("\n  Reference contrast (needs R_evar_base / R_evar_linear):")
    for cell in ('R_evar_base', 'R_evar_linear'):
        for arch, alab in ARCHS:
            r = by.get((cell, arch))
            if r is None:
                print(f"    {cell:15s} {alab:9s} NOT DOWNLOADED YET")
            else:
                print(f"    {cell:15s} {alab:9s} peakiness {r['peakiness']:.4f} "
                      f"= {r['over_sharpening']:.2f}x target   |   "
                      f"normalised loss {r['norm_loss']:.3f}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/flatevar')
    ap.add_argument('--only', nargs='+', default=None,
                    choices=['core', 'neural', 'sweep', 'sweeplin', 'metrics'])
    a = ap.parse_args()

    jobs = {'core': fig_core, 'neural': fig_neural_pcs,
            'sweep': fig_sweep_h8, 'sweeplin': fig_sweep_linear}
    want = a.only or (list(jobs) + ['metrics'])

    print(f"flatevar_v1 report -> {a.out_root}")
    for name in [j for j in jobs if j in want]:
        jobs[name](a.results_root, a.out_root)
    if 'metrics' in want:
        rows = write_metrics(a.results_root, a.out_root)
        if rows:
            print_headline(rows)


if __name__ == '__main__':
    main()
