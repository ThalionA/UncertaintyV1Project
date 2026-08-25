# -*- coding: utf-8 -*-
"""Pins for the canonical row-wise distribution metrics in
``decoder_metrics`` (2026-08-25 audit, item D1).

Two eps conventions live here and they are DIFFERENT METRICS, not rounding
variants — for peaked predictions against broad targets (this project's
over-sharpening regime) they disagree by roughly a factor of two:

  * ``eps_mode='clip'`` (default) — log(max(p, 1e-12)), weighting by the
    clipped value. What all seven replaced per-script copies did, hence what
    every KL number now in the figures/CSVs/vault was computed with.
  * ``eps_mode='additive'`` — log(p + float32 eps), weighting by the raw
    value. What the TRAINING side does, and it saturates (~15.9 nats max per
    confidently-wrong bin).

These tests pin each mode against its own reference, and pin the gap between
them so a future "let's just use one eps" tidy-up fails loudly instead of
silently halving published KLs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NN_DECODER = os.path.join(REPO_ROOT, 'nn_decoder')
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

from decoder_metrics import (  # noqa: E402
    cross_entropy_rows, entropy_rows, kl_rows, js_rows,
    peakiness_rows, mean_sem, EPS_CLIP, EPS_ADDITIVE,
)


def _rand_rows(n=13, bins=91, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, bins)) + 1e-3
    return x / x.sum(1, keepdims=True)


def _peaked_rows(n=13, bins=91, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.integers(10, bins - 10, (n, 1))
    g = np.exp(-0.5 * ((np.arange(bins)[None, :] - centres) / sigma) ** 2)
    return g / g.sum(1, keepdims=True)


def _kl_replaced_copy(a, b):
    """Verbatim code of the seven per-script copies this module replaced:
    KL(a || b) with clip-regularisation and clipped weighting."""
    a = np.clip(a, 1e-12, None)
    b = np.clip(b, 1e-12, None)
    return (a * (np.log(a) - np.log(b))).sum(-1)


# ----------------------------------------------------------------------
# 'clip' (default) reproduces the replaced copies — the consolidation must
# not have moved any number that is already in a figure.
# ----------------------------------------------------------------------

@pytest.mark.parametrize('case', ['smooth', 'peaked', 'zeros_target',
                                  'zeros_pred', 'zeros_both'])
def test_clip_mode_reproduces_the_replaced_copies(case):
    target, pred = _rand_rows(seed=1), _rand_rows(seed=2)
    if case == 'peaked':
        pred = _peaked_rows(seed=3)
    if case in ('zeros_target', 'zeros_both'):
        target = target.copy(); target[:, :30] = 0.0
        target /= target.sum(1, keepdims=True)
    if case in ('zeros_pred', 'zeros_both'):
        pred = pred.copy(); pred[:, :40] = 0.0
        pred /= pred.sum(1, keepdims=True)
    # NB argument order: the copies took (target, pred); kl_rows takes (pred, target).
    np.testing.assert_allclose(
        kl_rows(pred, target), _kl_replaced_copy(target, pred), atol=1e-12)


# ----------------------------------------------------------------------
# 'additive' reproduces the training-side torch implementations.
# ----------------------------------------------------------------------

def test_additive_mode_agrees_with_training_side_torch():
    torch = pytest.importorskip('torch')
    from nn_classifier import cross_entropy, entropy_calc, KL_calc, JS_calc
    p, q = _peaked_rows(seed=4), _rand_rows(seed=5)
    tp, tq = torch.tensor(p), torch.tensor(q)
    np.testing.assert_allclose(
        cross_entropy_rows(p, q, 'additive'), cross_entropy(tp, tq).numpy(), atol=1e-12)
    np.testing.assert_allclose(
        entropy_rows(p, 'additive'), entropy_calc(tp).numpy(), atol=1e-12)
    np.testing.assert_allclose(
        kl_rows(p, q, 'additive'), KL_calc(tp, tq).numpy(), atol=1e-12)
    np.testing.assert_allclose(
        js_rows(p, q, 'additive'), JS_calc(tp, tq).numpy(), atol=1e-12)


def test_additive_mode_agrees_with_fit_loss_per_trial():
    torch = pytest.importorskip('torch')
    from nn_classifier import fit_loss_per_trial
    p, q = _rand_rows(seed=6), _rand_rows(seed=7)
    ref = fit_loss_per_trial(torch.tensor(p), torch.tensor(q), 'KL').numpy()
    np.testing.assert_allclose(kl_rows(p, q, 'additive'), ref, atol=1e-12)


# ----------------------------------------------------------------------
# The conventions genuinely disagree — pin it so nobody "unifies" them.
# ----------------------------------------------------------------------

def test_the_two_conventions_disagree_substantially_when_predictions_are_sharp():
    target, pred = _rand_rows(n=300, seed=8), _peaked_rows(n=300, sigma=2.0, seed=9)
    clip_kl = kl_rows(pred, target, 'clip').mean()
    add_kl = kl_rows(pred, target, 'additive').mean()
    # Additive eps saturates the log, so it reports a MUCH smaller KL.
    assert add_kl < clip_kl
    assert clip_kl / add_kl > 1.5, (
        f"clip/additive ratio {clip_kl / add_kl:.2f} — the saturation gap this "
        f"test guards has changed; check nn_classifier's eps before editing.")
    # ...while for smooth, well-supported distributions they nearly agree.
    smooth = _rand_rows(n=300, seed=10)
    c, a = (kl_rows(smooth, target, m).mean() for m in ('clip', 'additive'))
    assert abs(c - a) / c < 1e-3


def test_eps_constants_match_their_sources():
    assert EPS_CLIP == 1e-12
    assert EPS_ADDITIVE == float(np.finfo(np.float32).eps)


def test_unknown_eps_mode_raises():
    p = _rand_rows(seed=11)
    with pytest.raises(ValueError, match='eps_mode'):
        kl_rows(p, p, 'clamp')


# ----------------------------------------------------------------------
# Identities and the small shared helpers.
# ----------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['clip', 'additive'])
def test_identities(mode):
    p = _rand_rows(seed=12)
    bins = p.shape[1]
    assert np.all(kl_rows(p, p, mode) == 0.0)          # clamped self-divergence
    np.testing.assert_allclose(js_rows(p, p, mode), 0.0, atol=1e-12)
    q = _rand_rows(seed=13)
    assert np.all(kl_rows(p, q, mode) >= 0)
    np.testing.assert_allclose(
        js_rows(p, q, mode), js_rows(q, p, mode), rtol=1e-12)
    u = np.full((1, bins), 1.0 / bins)
    np.testing.assert_allclose(entropy_rows(u, mode), np.log(bins), rtol=1e-4)
    np.testing.assert_allclose(peakiness_rows(u), 1.0 / bins)


def test_peakiness_matches_the_story_figures_primitive():
    p = _peaked_rows(seed=14)
    np.testing.assert_allclose(peakiness_rows(p), p.max(1))
    # story_figures.peaky is this, meaned over trials.
    np.testing.assert_allclose(peakiness_rows(p).mean(), p.max(1).mean())


def test_mean_sem_is_ddof1_and_nan_tolerant():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mean, sem = mean_sem(x)
    assert mean == pytest.approx(3.5)
    assert sem == pytest.approx(np.std(x, ddof=1) / np.sqrt(6))
    x_nan = np.array([1.0, 2.0, 3.0, np.nan])
    mean, sem = mean_sem(x_nan)
    assert mean == pytest.approx(2.0)
    assert sem == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3))
