# -*- coding: utf-8 -*-
"""Two-axis assessment of every `flatevar_v1` cell that survived the weight-decay
annihilation, as a single figure.

House rule: a decoder is judged on the BIAS axis and on performance under BOTH
scoring metrics — never one metric alone.
  * over-sharpening = decoded peakiness / IO target        (bias; 1.0 = on target)
  * performance under KL   = KL(decoded||target) / LOO predict-mean
  * performance under PROJECTION = the same ratio scored with the projection
    distance instead of KL, using a COMMON evar weighting for every decoder

The common basis matters. Each cell stores the weighting it was TRAINED with, so
scoring a cell with its own `explained_var` scores a flat-weighted decoder under
uniform weights (1/91 each) and an evar-weighted decoder under [0.907, 0.084, …].
Those are different metrics and their magnitudes are not comparable across rows.
Column c therefore rescoress every decoder under the SAME (evar) projection
weighting, taken from the evar baseline, so the column is a like-for-like
comparison. The PC basis itself is common already: it is fit on the same targets,
mice and split in every cell — only the weights differ.
Both metrics are shown because THEY DISAGREE, and the disagreement is the result:
the projection loss is blind to its own failure mode, so a projection-trained
decoder can look fine under its own metric while being far worse than chance
under KL. Scoring only under KL hides that the metrics diverge; scoring only
under the projection metric hides the failure entirely.
A decoder that lands the bias axis but fails on performance is a lobotomy.

Every point is one mouse (n=6), because the robust statement in this project is
sign consistency across animals, not the group mean — so the reader should be
able to count the animals rather than take a mean on trust.

The wd-matched evar baseline is imported from `hpsweep_v2`, which is the only run
containing an evar-weighted PCA cell at weight_decay=0 (the flat cells only
survive at wd=0, so comparing them against the wd=1e-4 baseline confounds the
weighting with the decay). Caveat carried in the axis label: hpsweep_v2 predates
the 2026-07-16 restart-selection fix.

Usage:  python diagnostics/flatevar_assessment.py
"""

from __future__ import annotations

import argparse
import glob
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
from cross_loss_eval import _eval_one  # noqa: E402  (same loss maths as training)

SPLIT = 'stratified_balanced'
HPSWEEP_WD0 = ('hpsweep_v2/lam0p003_drop0_acttanh_h8_pat0_vf0p2_wd0_shp0/'
               'Q_PCA_half_100ms_all')

# (label, results-relative path or flatevar cell, colour, marker)
ROWS = [
    ('KL  (H=8, wd 1e-4)',        ('flatevar_v1', 'R_reference_kl'),  ps.KL),
    ('JS  (H=8, wd 1e-4)',        ('flatevar_v1', 'R_reference_js'),  ps.JS),
    ('FLAT  (H=8, wd 0)',         ('flatevar_v1', 'A_flat_wd0'),      ps.FLAT_EVAR),
    ('FLAT  (linear, wd 0)',      ('flatevar_v1', 'B_flat_lin_wd0'),  ps.FLAT_EVAR),
    ('evar  (H=8, wd 0)  matched', ('RAW', HPSWEEP_WD0),              ps.PCA_EVAR),
    ('evar  (H=8, wd 1e-4)',      ('flatevar_v1', 'R_evar_base'),     ps.PCA_EVAR),
    ('evar + input PCA k=16',     ('flatevar_v1', 'C_evar_npc16'),    ps.PCA_EVAR),
]


def _path(results_root, spec):
    kind, val = spec
    if kind == 'RAW':
        return f'{results_root}/{val}/{SPLIT}.mat'
    hits = glob.glob(f'{results_root}/{kind}/{val}/*/{SPLIT}.mat')
    return hits[0] if hits else None


def _common_basis(results_root, arch):
    """(pcs, evar) per mouse from the evar baseline — the shared projection metric."""
    r = sio.loadmat(_path(results_root, ('flatevar_v1', 'R_evar_base')),
                    simplify_cells=True)['results']
    out = {}
    for m in r:
        if isinstance(r[m], dict) and isinstance(r[m].get('Dist'), dict):
            out[m] = (r[m]['Dist'].get('pcs'), r[m]['Dist'].get('explained_var'))
    return out


def _proj_common(r, arch, basis):
    """Normalised projection loss under the SHARED evar weighting, LOO predict-mean."""
    vals = []
    for m in sorted(r):
        if not (isinstance(r[m], dict) and isinstance(r[m].get('Dist'), dict)):
            continue
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


def load(path, arch, basis=None):
    r = sio.loadmat(path, simplify_cells=True)['results']
    mice = sorted(k for k in r if isinstance(r[k], dict)
                  and isinstance(r[k].get('Dist'), dict))
    pk = np.array([np.asarray(r[m]['Dist'][arch]['decoded'], float).max(1).mean()
                   for m in mice])
    tg = np.array([np.asarray(r[m]['Dist'][arch]['target'], float).max(1).mean()
                   for m in mice])
    norm = _norm_by_mouse(r, arch)
    proj = (_proj_common(r, arch, basis) if basis is not None
            else np.array(norm[('PCA', 'pm')], float))
    return pk / tg, np.array(norm[('KL', 'pm')], float), proj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/flatevar')
    a = ap.parse_args()

    ps.apply()
    # House style: categories on x, VERTICAL error bars. Rows = metric, cols = arch.
    fig, axes = plt.subplots(3, 2, figsize=ps.figsize(2, 3), sharex=True)
    x = np.arange(len(ROWS))
    METRICS = [
        ('bias', 'over-sharpening\n(peakiness / IO target)', 'IO target'),
        ('kl',   'normalised loss under KL\n(calibrated metric)', 'chance'),
        ('pca',  'normalised loss under PROJECTION\n(common evar weighting)', 'chance'),
    ]

    for ci, (arch, alab) in enumerate((('spat', 'spatial'), ('temp', 'temporal'))):
        basis = _common_basis(a.results_root, arch)
        cache = {}
        for lab, spec, colr in ROWS:
            p = _path(a.results_root, spec)
            if p is not None:
                cache[lab] = load(p, arch, basis)
        for ri, (metric, ylab, refl) in enumerate(METRICS):
            ax = axes[ri][ci]
            for xi, (lab, spec, colr) in zip(x, ROWS):
                if lab not in cache:
                    continue
                ratio, nl_kl, nl_pca = cache[lab]
                v = {'bias': ratio, 'kl': nl_kl, 'pca': nl_pca}[metric]
                jit = np.linspace(-0.17, 0.17, v.size)
                ax.plot(xi + jit, v, 'o', ms=3.5, mfc='none', mec=colr, mew=1.0,
                        alpha=0.85, ls='none', zorder=2)
                m = v.mean()
                sem = v.std(ddof=1) / np.sqrt(v.size)
                ax.errorbar(xi, m, yerr=sem, fmt='D', ms=6.5, color=colr,
                            mec='k', mew=0.6, capsize=4, lw=2.0, zorder=4)
            ax.axhline(1.0, color='k', ls=':', lw=1.4, zorder=1)
            ax.set_yscale('log')
            ax.grid(axis='y', alpha=0.25, lw=0.5)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if ri == 0:
                ax.set_title(alab, fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([r[0] for r in ROWS], rotation=38, ha='right',
                               fontsize=7)
            ax.set_xlim(-0.6, len(ROWS) - 0.4)

    axes[0][1].legend(handles=[
        Line2D([0], [0], marker='o', mfc='none', mec='0.35', ls='none', ms=4),
        Line2D([0], [0], marker='D', color='0.35', ls='none', ms=6),
        Line2D([0], [0], color='k', ls=':', lw=1.4),
    ], labels=['one mouse', 'mean ± sem', 'on target / chance'],
        fontsize=6.5, frameon=True, loc='best')

    ps.label_panels(axes.ravel())
    fig.suptitle('flatevar_v1 — every surviving decoder, n=6 mice, on all three axes. '
                 'Projection row uses one COMMON evar weighting so decoders are comparable. '
                 'Rows 2 and 3 disagree: the projection metric is blind to what KL exposes.',
                 y=1.01, fontsize=8.5)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'flatevar_fig5_assessment')


if __name__ == '__main__':
    main()
