# -*- coding: utf-8 -*-
"""Synthetic-data tests for the functional-alignment loadings comparison.

The five planted scenarios:
  1. Identity                — align decoder to itself; permutation must
                                 be identity and aligned-pair input
                                 cosine must be exactly 1.
  2. Permutation recovery    — shuffle B's hidden units by a known pi;
                                 alignment must recover pi^-1 and the
                                 aligned-pair input cosine must be 1.
  3. Sign-flip recovery      — flip the sign of a random subset of B's
                                 units (jointly on W_in and W_out, which
                                 preserves the network function); the
                                 alignment must recover the signs and
                                 aligned-pair input cosine must be 1.
  4. Independent random null — independent random W_in_A, W_in_B with
                                 H=32, n_neurons=500; aligned-pair input
                                 cosine distribution centred near zero
                                 (within sampling error).
  5. Planted shared subspace — W_in_A and W_in_B constructed to share a
                                 10-dim subspace; aligned-pair input
                                 cosine distribution noticeably above
                                 the random null.

These guard the alignment primitive without requiring trained models on
disk. The Hungarian alignment uses scipy.optimize.linear_sum_assignment;
scipy is required to run this file.
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

from decoder_loadings_comparison import (  # noqa: E402
    hungarian_align_units,
    aligned_pair_input_cosines,
    aligned_pair_output_cosines,
    compare_decoders,
)


# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------


def _random_decoder(rng, H=32, n_neurons=500, n_output=91, scale_in=1.0,
                     scale_out=1.0):
    """A pair (W_in, W_out) with independent random Gaussian entries.

    Shapes match SimpleFlexibleNNClassifier's single-hidden-layer
    backbone: W_in = (H, n_neurons); W_out = (n_output, H).
    """
    W_in = rng.normal(scale=scale_in, size=(H, n_neurons))
    W_out = rng.normal(scale=scale_out, size=(n_output, H))
    return W_in, W_out


# ----------------------------------------------------------------------
# 1. Identity
# ----------------------------------------------------------------------


def test_identity_recovers_identity_permutation_and_cos_one():
    rng = np.random.default_rng(0)
    W_in, W_out = _random_decoder(rng)
    row_ind, pi, signs = hungarian_align_units(W_out, W_out)
    assert np.array_equal(pi, np.arange(len(pi)))
    assert np.all(signs == 1)
    cos_in = aligned_pair_input_cosines(W_in, W_in, pi, signs)
    cos_out = aligned_pair_output_cosines(W_out, W_out, pi, signs)
    assert np.allclose(cos_in, 1.0)
    assert np.allclose(cos_out, 1.0)


# ----------------------------------------------------------------------
# 2. Permutation recovery
# ----------------------------------------------------------------------


def test_permutation_recovery_recovers_known_perm_and_cos_one():
    rng = np.random.default_rng(1)
    H = 32
    W_in_A, W_out_A = _random_decoder(rng, H=H)
    perm = rng.permutation(H)
    # B is A with hidden units permuted: W_in_B[k] = W_in_A[perm[k]],
    # W_out_B[:, k] = W_out_A[:, perm[k]].
    W_in_B = W_in_A[perm]
    W_out_B = W_out_A[:, perm]
    _, pi, signs = hungarian_align_units(W_out_A, W_out_B)
    # pi maps each A-unit k to the B-unit that matches it. Since B's
    # unit j is A's unit perm[j], B's unit that matches A's unit k is
    # the j with perm[j] == k -- i.e. pi = inverse_perm = argsort(perm).
    inv_perm = np.argsort(perm)
    assert np.array_equal(pi, inv_perm), (
        f"alignment should recover the INVERSE permutation; "
        f"pi={pi[:5]} inv_perm={inv_perm[:5]}"
    )
    assert np.all(signs == 1)
    cos_in = aligned_pair_input_cosines(W_in_A, W_in_B, pi, signs)
    assert np.allclose(cos_in, 1.0, atol=1e-8)


# ----------------------------------------------------------------------
# 3. Sign-flip recovery
# ----------------------------------------------------------------------


def test_sign_flip_recovery_recovers_signs_and_cos_one():
    rng = np.random.default_rng(2)
    H = 32
    W_in_A, W_out_A = _random_decoder(rng, H=H)
    # Flip sign of a random subset of A's units (jointly on W_in row and
    # W_out column) — this is the tanh symmetry: the network function is
    # unchanged. B = A with these joint flips.
    flips = rng.choice([-1.0, 1.0], size=H)
    W_in_B = (W_in_A.T * flips).T   # multiply rows of W_in by flips
    W_out_B = W_out_A * flips       # multiply columns of W_out by flips
    _, pi, signs = hungarian_align_units(W_out_A, W_out_B)
    # The recovered signs must equal flips (since pi is identity here).
    assert np.array_equal(pi, np.arange(H))
    assert np.array_equal(signs, flips)
    cos_in = aligned_pair_input_cosines(W_in_A, W_in_B, pi, signs)
    assert np.allclose(cos_in, 1.0, atol=1e-8)


def test_combined_perm_and_sign_recovery():
    """The combined case — both permutation and sign flips — must also
    recover correctly. The recovered signs are sign-of-B's-units-
    *in-pi-order*, so signs[k] == flips[pi[k]]."""
    rng = np.random.default_rng(3)
    H = 32
    W_in_A, W_out_A = _random_decoder(rng, H=H)
    perm = rng.permutation(H)
    flips = rng.choice([-1.0, 1.0], size=H)
    # Apply: B[k] = flips[k] * A[perm[k]]
    W_in_B = (W_in_A[perm].T * flips).T
    W_out_B = W_out_A[:, perm] * flips
    _, pi, signs = hungarian_align_units(W_out_A, W_out_B)
    inv_perm = np.argsort(perm)
    assert np.array_equal(pi, inv_perm)
    # signs[k] is the sign of B's pi[k]-th unit, which carries flips[pi[k]].
    assert np.array_equal(signs, flips[pi])
    cos_in = aligned_pair_input_cosines(W_in_A, W_in_B, pi, signs)
    assert np.allclose(cos_in, 1.0, atol=1e-8)


# ----------------------------------------------------------------------
# 4. Independent random null
# ----------------------------------------------------------------------


def test_independent_random_yields_null_cos_distribution():
    """With H=32 independent units in each of two networks, aligned-pair
    W_in cosines should be at chance — concentrated near zero. The
    Hungarian alignment maximises |cos(W_out)|, but since W_in is
    independent of W_out by construction here, the alignment carries
    no information about W_in, and post-alignment W_in cosines are
    distributed like cos of two independent random unit vectors in
    R^n_neurons (centred at 0 with std ~ 1/sqrt(n_neurons))."""
    rng = np.random.default_rng(4)
    n_neurons = 500
    H = 32
    W_in_A, W_out_A = _random_decoder(rng, H=H, n_neurons=n_neurons)
    W_in_B, W_out_B = _random_decoder(rng, H=H, n_neurons=n_neurons)
    _, pi, signs = hungarian_align_units(W_out_A, W_out_B)
    cos_in = aligned_pair_input_cosines(W_in_A, W_in_B, pi, signs)
    # Expected scale: |cos| of two random unit vectors in R^500 has
    # std ~ 1/sqrt(500) ~ 0.045. Mean |cos_in| of 32 such pairs should
    # be << 0.2.
    assert abs(np.mean(cos_in)) < 0.05, (
        f"Mean aligned-pair cos under independence should be ~0; "
        f"got {np.mean(cos_in):.3f}"
    )
    assert np.mean(np.abs(cos_in)) < 0.2, (
        f"Mean |aligned-pair cos| under independence should be small; "
        f"got {np.mean(np.abs(cos_in)):.3f}"
    )


# ----------------------------------------------------------------------
# 5. Planted shared subspace
# ----------------------------------------------------------------------


def test_planted_shared_subspace_lifts_aligned_cosine_above_null():
    """Both decoders' W_in rows are drawn from a shared 10-dim subspace
    (so input cosines between any pair of A-row and B-row are
    structurally higher than chance). The Hungarian aligns on W_out as
    usual; aligned-pair input cosines should land above the
    independent-random null."""
    rng = np.random.default_rng(5)
    n_neurons = 500
    H = 32
    shared_dim = 10
    # Shared orthonormal subspace basis
    U = rng.normal(size=(n_neurons, shared_dim))
    U, _ = np.linalg.qr(U)  # orthonormalise columns
    # W_in_A and W_in_B each have rows drawn as a random combination of
    # the shared columns plus a small random noise.
    coefs_A = rng.normal(size=(H, shared_dim))
    coefs_B = rng.normal(size=(H, shared_dim))
    noise_A = 0.05 * rng.normal(size=(H, n_neurons))
    noise_B = 0.05 * rng.normal(size=(H, n_neurons))
    W_in_A = coefs_A @ U.T + noise_A
    W_in_B = coefs_B @ U.T + noise_B
    # W_out independent random (so alignment is purely output-driven)
    W_out_A = rng.normal(size=(91, H))
    W_out_B = rng.normal(size=(91, H))
    _, pi, signs = hungarian_align_units(W_out_A, W_out_B)
    cos_in = aligned_pair_input_cosines(W_in_A, W_in_B, pi, signs)
    # The shared subspace pumps any pair of W_in rows from A and B
    # toward a non-zero cosine of roughly 1/sqrt(shared_dim) ~ 0.32 in
    # absolute value (because random vectors restricted to a d-dim
    # subspace have |cos| ~ 1/sqrt(d) with each other). Hungarian on
    # W_out doesn't help here (W_out is independent of W_in), so the
    # aligned-pair |cos| distribution should match the population
    # |cos| of random W_in pairs in the shared subspace — which is
    # well above the n_neurons=500 chance level of 1/sqrt(500) ~ 0.045.
    assert np.mean(np.abs(cos_in)) > 0.15, (
        f"Planted shared subspace should lift mean |cos| above chance; "
        f"got {np.mean(np.abs(cos_in)):.3f}"
    )


# ----------------------------------------------------------------------
# Wrapper: compare_decoders contract
# ----------------------------------------------------------------------


def test_compare_decoders_returns_expected_keys():
    rng = np.random.default_rng(6)
    W_in_A, W_out_A = _random_decoder(rng, H=8, n_neurons=20, n_output=5)
    W_in_B, W_out_B = _random_decoder(rng, H=8, n_neurons=20, n_output=5)
    r = compare_decoders(W_in_A, W_out_A, W_in_B, W_out_B)
    for key in ('pi', 'signs', 'cos_out_aligned', 'cos_in_aligned',
                 'cos_in_raw', 'cos_out_matrix_pre'):
        assert key in r, f"compare_decoders missing key {key!r}"
    assert r['cos_out_matrix_pre'].shape == (8, 8)
    assert r['cos_in_aligned'].shape == (8,)
    assert r['cos_in_raw'].shape == (8,)


def test_compare_decoders_identity_gives_high_aligned_cos():
    rng = np.random.default_rng(7)
    W_in, W_out = _random_decoder(rng)
    r = compare_decoders(W_in, W_out, W_in, W_out)
    assert np.mean(r['cos_in_aligned']) > 0.99
