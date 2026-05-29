# -*- coding: utf-8 -*-
"""Regression tests for the PCA-vs-KL/JS smoothness demo.

Pins the two headline facts of ``nn_decoder/diagnostics/loss_smoothness_demo.py``
(see ``figures/loss_smoothness_demo/LOSS_SMOOTHNESS_REPORT.md``):

  1. Temporal mixture: as per-bin posteriors sharpen, forward KL suffers far
     more than JS, while PCA does not suffer at all (KL ≫ JS ≫ PCA).
  2. Direct fit: PCA exerts almost no force to broaden an over-confident
     posterior, whereas KL and JS broaden it back toward the target.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIAG = os.path.join(REPO_ROOT, 'nn_decoder', 'diagnostics')
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
for p in (NN_DECODER, DIAG):
    if p not in sys.path:
        sys.path.insert(0, p)

from loss_smoothness_demo import (  # noqa: E402
    N_CATS, fit_basis, bump, mixture_average, all_losses, optimise_single,
    entropy_np,
)

BROAD_SIGMA = 9.0


def _mixture_curves():
    pcs, evar, _ = fit_basis(BROAD_SIGMA)
    target = bump(N_CATS // 2, BROAD_SIGMA)
    cdf = np.cumsum(target)
    T = 12
    centers = np.interp((np.arange(T) + 0.5) / T, cdf, np.arange(N_CATS))
    sharp = all_losses(mixture_average(centers, 0.6)[1], target, pcs, evar)
    broad = all_losses(mixture_average(centers, BROAD_SIGMA * 1.4)[1], target, pcs, evar)
    return sharp, broad


def test_temporal_mixture_kl_suffers_more_than_js_more_than_pca():
    sharp, broad = _mixture_curves()
    kl_frac = (sharp['KL'] - broad['KL']) / broad['KL']
    js_frac = (sharp['JS'] - broad['JS']) / broad['JS']
    pca_frac = (sharp['PCA'] - broad['PCA']) / broad['PCA']

    # KL suffers a lot (more than doubles), JS less, PCA not at all.
    assert kl_frac > 1.0, f"KL should blow up on the gappy average; got {kl_frac:.2f}"
    assert kl_frac > js_frac > pca_frac, (
        f"expected KL>JS>PCA fractional increase; "
        f"got KL={kl_frac:.2f} JS={js_frac:.2f} PCA={pca_frac:.2f}")
    assert pca_frac < 0.1, f"PCA should barely move; got {pca_frac:.2f}"


def test_pca_does_not_broaden_an_overconfident_posterior():
    pcs, evar, _ = fit_basis(BROAD_SIGMA)
    pcs_t, evar_t = torch.tensor(pcs), torch.tensor(evar)
    target = bump(N_CATS // 2, BROAD_SIGMA)
    target_H = entropy_np(target)
    init = torch.tensor(np.log(bump(N_CATS // 2, 1.2) + 1e-12), dtype=torch.float32)

    H = {}
    for name in ('PCA', 'KL', 'JS'):
        pred, _, _ = optimise_single(target, name, pcs_t, evar_t, init,
                                     steps=1500, lr=0.05)
        H[name] = entropy_np(pred)

    # KL/JS broaden the spike toward the target; PCA stays peaky.
    assert H['PCA'] < target_H - 1.0, f"PCA should stay peaky; H={H['PCA']:.2f}"
    assert H['KL'] > H['PCA'] + 1.0, f"KL should broaden; KL={H['KL']:.2f} PCA={H['PCA']:.2f}"
    assert H['JS'] > H['PCA'], f"JS should broaden more than PCA; JS={H['JS']:.2f} PCA={H['PCA']:.2f}"
