# -*- coding: utf-8 -*-
"""Why does lambda_H flatten the flat/MSE temporal decoder? — the diagnosis.

Surface reading of `projflat_fig3`: raising the SBC entropy penalty lambda_H makes
the decoded temporal posterior BROADER (peaky/target 0.90 -> 0.40 -> 0.33). That
is counter-intuitive: the entropy penalty is a SHARPNESS penalty
(`total = fit_loss + lambda_H * mean(H(per-bin)))`, `entropy_calc = -sum p log p`,
correctly signed), so it should sharpen, not flatten. It looks like a bug.

It is NOT a sign bug. It is a LOSS-SCALE MISMATCH, the same family as the
weight-decay annihilation. Three facts, shown here:

  a/b  Under flat/MSE the fit-loss is tiny (~3e-4: MSE between two near-flat
       91-bin distributions). The penalty is lambda_H * H with H ~ 4.5, i.e.
       ~0.0135 at lambda_H=3e-3 — about 45x the fit-loss (150x at 1e-2). So the
       total loss is almost entirely the entropy term; from epoch 0 the optimiser
       trades the fit away, val fit-loss RISES, and early-stopping (patience 20,
       selection on val fit-loss) bails at ~epoch 20 to a near-initialisation
       (uniform) model.

  c    The clincher: the SAME penalty at patience 0 (flatevar_v1, no early stop)
       runs to completion and does exactly what it should — per-bin entropy 0.90
       (near-delta bins), which average to an OVER-sharpened posterior. patience
       20 gives per-bin entropy 4.43 (near uniform). Same lambda_H, opposite
       per-bin sharpness, entirely explained by when training stops.

So the penalty sharpens correctly; the flat posteriors at patience 20 are
under-training, not entropy-maximisation. lambda_H under flat/MSE would need
scaling by the fit-loss magnitude to be a meaningful regulariser (cf. the
smoothness-lambda fit-loss scaling, GOTCHAS 2026-06-16).

Usage:  python diagnostics/projflat_lambda_diagnosis.py
"""

from __future__ import annotations

import argparse
import glob
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

LOGN = np.log(91)
CELLS = [('λ_H=0', 'projflat_v1', 'h8_raw_l0_d0_w0'),
         ('λ_H=3e-3', 'projflat_v1', 'h8_raw_l0p003_d0_w0'),
         ('λ_H=1e-2', 'projflat_v1', 'h8_raw_l0p01_d0_w0')]


def _mat(results_root, run, cell):
    hits = glob.glob(f'{results_root}/{run}/{cell}/*/stratified_balanced.mat')
    return hits[0] if hits else None


def _perbin_H(results_root, run, cell):
    p = _mat(results_root, run, cell)
    if p is None:
        return None
    r = sio.loadmat(p, simplify_cells=True)['results']
    hs = []
    for m in sorted(r):
        if not (isinstance(r[m], dict) and isinstance(r[m].get('Dist'), dict)):
            continue
        s = np.asarray(r[m]['Dist']['temp']['decoded_samp'], float)
        s = s[np.isfinite(s).all((1, 2))]
        sc = np.clip(s, 1e-12, None)
        sc = sc / sc.sum(1, keepdims=True)
        hs.append((-(sc * np.log(sc)).sum(1)).mean())
    return np.array(hs)


def _curve(results_root, run, cell, field):
    cks = sorted(glob.glob(f'{results_root}/{run}/{cell}/*/checkpoints/'
                           'mouse_0_stratified_balanced.pt'))
    if not cks:
        return None
    node = torch.load(cks[0], map_location='cpu', weights_only=False).get('temp')
    return np.array((node or {}).get('history', {}).get(field, []))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--out-root', default='figures/projflat')
    a = ap.parse_args()
    ps.apply()
    colr = {'λ_H=0': ps.KL, 'λ_H=3e-3': ps.SPATIAL, 'λ_H=1e-2': ps.PCA_EVAR}

    fig, axes = plt.subplots(1, 3, figsize=ps.figsize(3, 1))
    for field, ax, ttl in [('val_fit_loss', axes[0], 'val fit-loss — λ_H>0 rises from epoch 0'),
                           ('train_fit_loss', axes[1], 'train fit-loss — the penalty sacrifices the fit')]:
        for lab, run, cell in CELLS:
            y = _curve(a.results_root, run, cell, field)
            if y is None or not len(y):
                continue
            ax.plot(np.arange(len(y)), y, color=colr[lab], lw=1.6, label=lab)
            ax.plot(len(y) - 1, y[-1], 'o', color=colr[lab], ms=5)
        ax.set_xlabel('epoch')
        ax.set_ylabel('MSE fit-loss')
        ax.set_yscale('log')
        ax.set_title(ttl, fontsize=8)
        ax.legend(fontsize=7)

    # panel c — the clincher: same lambda_H=3e-3, patience 0 vs 20
    ax = axes[2]
    pairs = [('patience 0\n(no early stop)', 'flatevar_v1', 'A_flat_wd0', ps.PCA_EVAR),
             ('patience 20\n(early stop)', 'projflat_v1', 'h8_raw_l0p003_d0_w0', ps.SPATIAL)]
    for xi, (lab, run, cell, c) in enumerate(pairs):
        H = _perbin_H(a.results_root, run, cell)
        if H is None:
            continue
        jit = np.linspace(-0.12, 0.12, H.size)
        ax.plot(xi + jit, H, 'o', ms=4, mfc='none', mec=c, mew=1.0, ls='none')
        ax.errorbar(xi, H.mean(), yerr=H.std(ddof=1) / np.sqrt(H.size), fmt='D',
                    ms=7, color=c, mec='k', mew=0.6, capsize=4)
    ax.axhline(LOGN, color='k', ls=':', lw=1.2)
    ax.text(1.4, LOGN, 'uniform', fontsize=6.5, va='bottom', ha='right', color='0.4')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([p[0] for p in pairs], fontsize=7.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('per-bin entropy  H(per-bin posterior)')
    ax.set_title('SAME λ_H=3e-3: the penalty sharpens\nbins (patience 0); early-stop bails (20)',
                 fontsize=8)
    ps.label_panels(axes)
    fig.suptitle('λ_H does not broaden the temporal decoder — it is a loss-scale mismatch. The entropy penalty '
                 '(~λ_H·4.5) is 45–150× the tiny MSE fit-loss (~3e-4), so val fit-loss rises from epoch 0 and '
                 'early-stopping saves a near-init (uniform) model.  (c) proves the penalty DOES sharpen when '
                 'allowed to run.', y=1.05, fontsize=7.3)
    fig.tight_layout()
    ps.save_fig(fig, Path(a.out_root), 'projflat_fig4_lambdaH_diagnosis')


if __name__ == '__main__':
    main()
