# -*- coding: utf-8 -*-
"""Synthetic-data tests for ``nn_decoder/similarity_analysis.py``.

These tests cover:
  - ``_si_normalised``: bounded SI in [-1, +1], symmetry, fixed points.
  - ``_compute_archetypes_exemplars``: shape, content, consistency with the
    existing prototype-mean ``_compute_archetypes``.
  - ``_archetype_similarity_exemplars`` + ``_aggregate_exemplars``:
    expected-exemplar reduces to prototype in the noiseless limit;
    max-exemplar dominates expected-exemplar pointwise.
  - vMF helpers (``_kappa_banerjee``, ``_log_C_p``,
    ``_archetype_similarity_vmf``): known Banerjee values, log-normaliser
    sanity, posterior-form SI bounded in [-1, +1] and sign-correct on
    synthetic Go / NoGo trials.
  - The ``_si_per_method`` dispatcher: each variant returns finite,
    bounded SI of the right shape; in the noiseless limit, prototype and
    expected-exemplar give identical SI.

The synthetic mouse builder ``_make_mouse`` constructs a small
``MouseData`` with known orthogonal Go / NoGo archetype patterns,
clearly-easy trials (top contrast, low dispersion) and ambiguous trials
(low contrast, high dispersion). Tests using non-zero noise check
qualitative behaviour (sign of SI, ordering) rather than exact equality.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NN_DECODER = os.path.join(REPO_ROOT, "nn_decoder")
if NN_DECODER not in sys.path:
    sys.path.insert(0, NN_DECODER)

import similarity_analysis as dre  # noqa: E402 — aliased as ``dre`` to keep test bodies stable


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_mouse(
    n_easy_per_side: int = 30,
    n_ambig: int = 40,
    n_neurons: int = 48,
    n_xG: int = 30,
    *,
    seed: int = 0,
    noise_amp: float = 0.1,
    bundle_spread: float = 0.0,
) -> "dre.MouseData":
    """Build a synthetic MouseData with known orthogonal Go / NoGo archetypes.

    Each easy trial is ``archetype + structured per-trial perturbation +
    iid Gaussian noise``. Ambiguous trials are interpolations between
    archetypes with low contrast / high dispersion (so they fail the
    "easy" filter inside ``_compute_archetypes``).

    bundle_spread controls how much each exemplar's direction is perturbed
    from the bundle mean (useful for exemplar-variant tests).
    """
    rng = _rng(seed)

    # Two orthonormal archetype direction vectors in N-d neural space.
    base = rng.standard_normal((2, n_neurons))
    base[1] = base[1] - (base[1] @ base[0]) * base[0] / (base[0] @ base[0])
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    ramp = np.linspace(0.0, 1.0, n_xG)
    arch_go = base[0][:, None] * ramp[None, :]  # (n_neurons, n_xG)
    arch_nogo = base[1][:, None] * ramp[None, :]

    n_trials = 2 * n_easy_per_side + n_ambig
    activity = np.zeros((n_trials, n_neurons, n_xG))
    orientation = np.zeros(n_trials)
    contrast = np.zeros(n_trials)
    dispersion = np.zeros(n_trials)

    idx = 0
    # Easy Go trials
    for _ in range(n_easy_per_side):
        perturb = bundle_spread * rng.standard_normal(n_neurons)
        trial_arch = arch_go + perturb[:, None] * ramp[None, :]
        activity[idx] = trial_arch + noise_amp * rng.standard_normal((n_neurons, n_xG))
        orientation[idx] = 10.0  # < 45° → Go side
        contrast[idx] = 1.0
        dispersion[idx] = 1.0
        idx += 1
    # Easy NoGo trials
    for _ in range(n_easy_per_side):
        perturb = bundle_spread * rng.standard_normal(n_neurons)
        trial_arch = arch_nogo + perturb[:, None] * ramp[None, :]
        activity[idx] = trial_arch + noise_amp * rng.standard_normal((n_neurons, n_xG))
        orientation[idx] = 80.0  # > 45° → NoGo side
        contrast[idx] = 1.0
        dispersion[idx] = 1.0
        idx += 1
    # Ambiguous trials — fail the easy filter
    for _ in range(n_ambig):
        alpha = rng.uniform(0, 1)
        mix = alpha * arch_go + (1.0 - alpha) * arch_nogo
        activity[idx] = mix + noise_amp * rng.standard_normal((n_neurons, n_xG))
        orientation[idx] = 10.0 if alpha > 0.5 else 80.0
        contrast[idx] = 0.1
        dispersion[idx] = 10.0
        idx += 1

    xG = np.linspace(0.0, 2.0, n_xG)
    return dre.MouseData(
        tag="synthetic",
        activity=activity,
        xG=xG,
        orientation=orientation,
        true_orientation=np.where(orientation < dre.GO_BOUNDARY_DEG, 0.0, 90.0),
        contrast=contrast,
        dispersion=dispersion,
        choice=np.zeros(n_trials),
        velocity=np.zeros(n_trials),
        lick_rate=np.zeros(n_trials),
        post_s_marginal=np.zeros((n_trials, len(dre.S_GRID))),
        decision_posterior=np.zeros((n_trials, 2)),
    )


# ----------------------------------------------------------------------
# _si_normalised
# ----------------------------------------------------------------------


def test_si_normalised_bounds():
    rng = _rng(1)
    sR = rng.uniform(-1.0, 1.0, size=1000)
    sL = rng.uniform(-1.0, 1.0, size=1000)
    SI = dre._si_normalised(sR, sL)
    assert np.all(SI >= -1.0 - 1e-9)
    assert np.all(SI <= 1.0 + 1e-9)


def test_si_normalised_antisymmetric():
    rng = _rng(2)
    sR = rng.uniform(-1.0, 1.0, size=200)
    sL = rng.uniform(-1.0, 1.0, size=200)
    assert np.allclose(dre._si_normalised(sR, sL),
                       -dre._si_normalised(sL, sR), atol=1e-9)


def test_si_normalised_zero_when_equal():
    sR = np.array([-0.5, 0.0, 0.7])
    sL = sR.copy()
    assert np.allclose(dre._si_normalised(sR, sL), 0.0, atol=1e-9)


def test_si_normalised_extremes():
    one = np.array([1.0])
    minus = np.array([-1.0])
    assert np.isclose(dre._si_normalised(one, minus), 1.0, atol=1e-6)
    assert np.isclose(dre._si_normalised(minus, one), -1.0, atol=1e-6)


# ----------------------------------------------------------------------
# Exemplar archetypes
# ----------------------------------------------------------------------


def test_compute_archetypes_exemplars_shape():
    m = _make_mouse(n_easy_per_side=20)
    ex = dre._compute_archetypes_exemplars(m)
    assert set(ex.keys()) == {"Go", "NoGo"}
    K_go = ex["Go"].shape[0]
    K_nogo = ex["NoGo"].shape[0]
    assert K_go >= dre.ARCHETYPE_MIN_TRIALS
    assert K_nogo >= dre.ARCHETYPE_MIN_TRIALS
    assert ex["Go"].shape == (K_go, m.n_neurons, m.n_xG)
    assert ex["NoGo"].shape == (K_nogo, m.n_neurons, m.n_xG)


def test_compute_archetypes_exemplars_mean_matches_prototype():
    """Mean over exemplars == prototype trajectory (both in z-scored space)."""
    m = _make_mouse(seed=3)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    for side in proto:
        assert np.allclose(ex[side].mean(axis=0), proto[side], atol=1e-9)


def test_compute_archetypes_exemplars_min_trials():
    """A side with too few easy trials should be omitted.

    Note: ``_compute_archetypes_exemplars`` uses a *quantile* filter on
    contrast / dispersion, so the "easy" mask is sensitive to the overall
    distribution. We strip the ambiguous trials entirely so the per-side
    easy count equals ``n_easy_per_side``; below the minimum, the side is
    omitted.
    """
    m = _make_mouse(n_easy_per_side=3, n_ambig=0)
    assert m.n_trials == 6  # 3 easy Go + 3 easy NoGo, nothing else
    ex = dre._compute_archetypes_exemplars(m)
    assert "Go" not in ex
    assert "NoGo" not in ex


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def test_expected_exemplar_equals_prototype_in_noiseless_limit():
    """When all exemplars are identical (zero noise + zero bundle_spread),
    cosine-to-mean and mean-of-cosines must agree exactly because z-scoring
    is linear and all exemplars in the bundle reduce to the same trajectory.
    """
    m = _make_mouse(seed=4, noise_amp=0.0, bundle_spread=0.0)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    sim_proto, names_p = dre._archetype_similarity(m, proto)
    sim_ex, names_e = dre._archetype_similarity_exemplars(m, ex)
    sim_expected = dre._aggregate_exemplars(sim_ex, "expected")
    for i, name in enumerate(names_p):
        j = names_e.index(name)
        assert np.allclose(sim_proto[..., i], sim_expected[..., j], atol=1e-8)


def test_max_exemplar_dominates_expected():
    """max_k ≥ mean_k always (where both are finite)."""
    m = _make_mouse(seed=5, noise_amp=0.05, bundle_spread=0.5)
    ex = dre._compute_archetypes_exemplars(m)
    sim_ex, _ = dre._archetype_similarity_exemplars(m, ex)
    expected = dre._aggregate_exemplars(sim_ex, "expected")
    max_agg = dre._aggregate_exemplars(sim_ex, "max")
    valid = np.isfinite(expected) & np.isfinite(max_agg)
    assert np.all(max_agg[valid] + 1e-9 >= expected[valid])


def test_aggregate_unknown_method_raises():
    with pytest.raises(ValueError):
        dre._aggregate_exemplars(np.zeros((2, 3, 2, 4)), method="bogus")


# ----------------------------------------------------------------------
# vMF helpers
# ----------------------------------------------------------------------


def test_kappa_banerjee_known_value():
    # R_bar = 0.5, p = 10  →  kappa = 0.5 * (10 - 0.25) / (1 - 0.25) = 6.5
    kappa = dre._kappa_banerjee(np.array([0.5]), p=10)
    assert np.isclose(kappa[0], 6.5, atol=1e-6)


def test_kappa_banerjee_zero_bar():
    # R_bar = 0  →  kappa = 0 (uniform distribution)
    kappa = dre._kappa_banerjee(np.array([0.0]), p=20)
    assert np.isclose(kappa[0], 0.0, atol=1e-9)


def test_kappa_banerjee_high_bar_clips():
    # R_bar → 1 should not blow up due to the internal clip.
    kappa = dre._kappa_banerjee(np.array([1.0]), p=20)
    assert np.isfinite(kappa[0])
    assert kappa[0] > 1e6  # very tight bundle → very large kappa


def test_log_C_p_finite():
    # Across a sweep of kappas and a moderate p, log_C_p must be finite.
    kappa = np.array([0.01, 0.1, 1.0, 5.0, 50.0, 500.0])
    val = dre._log_C_p(kappa, p=64)
    assert np.all(np.isfinite(val))


def test_vmf_si_sign_correct_on_easy_trials():
    """Easy-Go-side trials should give SI > 0 on average; easy-NoGo SI < 0.

    Uses tanh((log p_R − log p_L) / 2) ∈ [-1, 1] via _si_per_method('vmf').
    """
    m = _make_mouse(seed=6, n_easy_per_side=25, noise_amp=0.05, bundle_spread=0.2)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    si_vmf = dre._si_per_method(m, proto, ex, method="vmf")
    assert si_vmf.shape == (m.n_trials, m.n_xG)
    assert np.all(si_vmf >= -1.0 - 1e-9)
    assert np.all(si_vmf <= 1.0 + 1e-9)
    is_easy_go = (m.contrast >= 0.5) & (m.orientation < dre.GO_BOUNDARY_DEG)
    is_easy_nogo = (m.contrast >= 0.5) & (m.orientation >= dre.GO_BOUNDARY_DEG)
    # Take the trial-average of the per-xG-mean SI for stability.
    mean_si_go = si_vmf[is_easy_go].mean()
    mean_si_nogo = si_vmf[is_easy_nogo].mean()
    assert mean_si_go > 0.05, f"easy-Go mean SI was {mean_si_go:.3f}, expected > 0.05"
    assert mean_si_nogo < -0.05, f"easy-NoGo mean SI was {mean_si_nogo:.3f}, expected < -0.05"


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------


def test_si_per_method_shapes_and_bounds():
    m = _make_mouse(seed=7, noise_amp=0.05, bundle_spread=0.1)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    for method in ("prototype", "expected_exemplar", "max_exemplar", "vmf"):
        si = dre._si_per_method(m, proto, ex, method=method)
        assert si.shape == (m.n_trials, m.n_xG), f"{method}: shape {si.shape}"
        assert np.all(np.isfinite(si)), f"{method}: non-finite values"
        assert np.all(si >= -1.0 - 1e-9), f"{method}: below -1"
        assert np.all(si <= 1.0 + 1e-9), f"{method}: above +1"


def test_si_per_method_prototype_equals_expected_in_noiseless_limit():
    m = _make_mouse(seed=8, noise_amp=0.0, bundle_spread=0.0)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    si_proto = dre._si_per_method(m, proto, ex, method="prototype")
    si_exp = dre._si_per_method(m, proto, ex, method="expected_exemplar")
    assert np.allclose(si_proto, si_exp, atol=1e-8)


def test_si_per_method_unknown_raises():
    m = _make_mouse(seed=9)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    with pytest.raises(ValueError):
        dre._si_per_method(m, proto, ex, method="not_a_method")


# ----------------------------------------------------------------------
# Per-animal framework metrics
# ----------------------------------------------------------------------


def test_per_animal_kappa_finite_and_per_side():
    m = _make_mouse(seed=20, noise_amp=0.05, bundle_spread=0.1)
    ex = dre._compute_archetypes_exemplars(m)
    kap = dre._per_animal_kappa(m, ex)
    assert set(kap.keys()) == {"Go", "NoGo"}
    for v in kap.values():
        assert np.isfinite(v) and v >= 0


def test_per_animal_kappa_decreases_with_bundle_spread():
    """Wider bundles → smaller κ. Same seeds so only spread varies."""
    m_tight = _make_mouse(seed=21, noise_amp=0.0, bundle_spread=0.0)
    m_loose = _make_mouse(seed=21, noise_amp=0.0, bundle_spread=1.0)
    k_t = dre._per_animal_kappa(m_tight, dre._compute_archetypes_exemplars(m_tight))
    k_l = dre._per_animal_kappa(m_loose, dre._compute_archetypes_exemplars(m_loose))
    for side in ("Go", "NoGo"):
        assert k_t[side] > k_l[side], (
            f"{side}: tight κ={k_t[side]:.2f} not > loose κ={k_l[side]:.2f}"
        )


def test_per_animal_kappa_empty_when_no_exemplars():
    m = _make_mouse(seed=22)
    assert dre._per_animal_kappa(m, {}) == {}


def test_per_animal_si_spread_finite():
    m = _make_mouse(seed=23, noise_amp=0.1)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    spread = dre._per_animal_si_spread(m, proto, ex, method="prototype")
    assert np.isfinite(spread) and spread >= 0


def test_si_methods_diverge_with_bundle_spread():
    """Prototype and expected-exemplar SI must differ once the bundle is broad.

    The framework's whole bundle / exemplar distinction relies on
    cosine-to-mean ≠ mean-of-cosines for non-degenerate bundles. With
    non-zero ``bundle_spread`` the two should produce visibly different
    $SI$ on some trials.
    """
    m = _make_mouse(seed=24, noise_amp=0.05, bundle_spread=0.5)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    si_proto = dre._si_per_method(m, proto, ex, method="prototype")
    si_exp = dre._si_per_method(m, proto, ex, method="expected_exemplar")
    # They should differ somewhere; equality would mean bundle width is
    # empirically vacuous under this aggregation.
    assert not np.allclose(si_proto, si_exp, atol=1e-3)


def test_decision_entropy_known_values():
    # p = 0.5 → max entropy = log(2)
    h = dre._decision_entropy(np.array([[0.5, 0.5]]))
    assert np.isclose(h[0], np.log(2.0), atol=1e-9)
    # p = 1 and p = 0 → entropy = 0
    assert np.isclose(dre._decision_entropy(np.array([[1.0, 0.0]]))[0], 0.0, atol=1e-9)
    assert np.isclose(dre._decision_entropy(np.array([[0.0, 1.0]]))[0], 0.0, atol=1e-9)


def test_decision_entropy_nan_propagation():
    dp = np.array([[np.nan, np.nan], [0.3, 0.7]])
    h = dre._decision_entropy(dp)
    assert np.isnan(h[0])
    assert np.isfinite(h[1])


def test_psychometric_perfect_observer():
    """If choice is deterministic from signed_contrast, P(Go) is 1 or 0 per side."""
    m = _make_mouse(seed=25, ambig_frac=0.0) if False else _make_mouse(seed=25)
    feats = dre.compute_features(m)
    c = feats["signed_contrast"]
    # Force a perfect observer: choice = 1 iff signed_contrast > 0.
    m.choice = np.where(c > 0, 1.0, 0.0)
    levels, p_go, n_per = dre._psychometric(m)
    assert len(levels) >= 2
    for lv, pg, n in zip(levels, p_go, n_per):
        if n == 0:
            continue
        if lv > 0:
            assert pg == 1.0
        elif lv < 0:
            assert pg == 0.0


def test_psychometric_shape_and_nan_at_empty_levels():
    m = _make_mouse(seed=26)
    levels, p_go, n_per = dre._psychometric(m)
    assert levels.shape == p_go.shape == n_per.shape
    for pg, n in zip(p_go, n_per):
        if n == 0:
            assert np.isnan(pg)
        else:
            assert 0.0 <= pg <= 1.0


# ----------------------------------------------------------------------
# Behavioural readouts (Q3-suite: lapse, velocity, logistic SI, DDM)
# ----------------------------------------------------------------------


def test_per_animal_lapse_perfect_observer_has_zero_lapse():
    """Choice = sign(signed_contrast) → no lapses on strong-evidence trials."""
    m = _make_mouse(seed=30, n_easy_per_side=30, n_ambig=20)
    feats = dre.compute_features(m)
    c = feats["signed_contrast"]
    m.choice = (c > 0).astype(float)
    lapse_go, lapse_nogo = dre._per_animal_lapse(m, contrast_threshold=0.5)
    assert lapse_go == 0.0
    assert lapse_nogo == 0.0


def test_per_animal_lapse_anti_observer_has_full_lapse():
    """Choice = sign(-signed_contrast) → lapse rate = 1.0 on both sides."""
    m = _make_mouse(seed=31, n_easy_per_side=30, n_ambig=20)
    feats = dre.compute_features(m)
    c = feats["signed_contrast"]
    m.choice = (c < 0).astype(float)
    lapse_go, lapse_nogo = dre._per_animal_lapse(m, contrast_threshold=0.5)
    assert lapse_go == 1.0
    assert lapse_nogo == 1.0


def test_per_animal_velocity_drift_recovers_planted_slope():
    """If velocity = α·signed_contrast + β + noise, recover α and β."""
    rng = _rng(32)
    m = _make_mouse(seed=32)
    feats = dre.compute_features(m)
    c = feats["signed_contrast"]
    true_alpha, true_bias = 2.0, 0.5
    m.velocity = true_alpha * c + true_bias + 0.05 * rng.standard_normal(m.n_trials)
    fit = dre._per_animal_velocity_drift(m)
    assert abs(fit["alpha"] - true_alpha) < 0.1
    assert abs(fit["bias"] - true_bias) < 0.1
    assert fit["r_squared"] > 0.95


def test_per_animal_si_logistic_returns_finite_likelihoods():
    m = _make_mouse(seed=33, noise_amp=0.05, n_easy_per_side=25)
    # Plant a choice ~ sign(signed_contrast)
    feats = dre.compute_features(m)
    c = feats["signed_contrast"]
    m.choice = (c > 0).astype(float)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    fit = dre._per_animal_si_logistic(m, proto, ex, method="prototype")
    assert np.isfinite(fit["ll_si"])
    assert np.isfinite(fit["ll_stim"])
    # Stimulus-only model should fit a near-perfect deterministic choice exactly,
    # so its LL should be very close to 0 (best possible) or at least better
    # than chance (n * log 0.5).
    chance_ll = -m.n_trials * np.log(2)
    assert fit["ll_stim"] > chance_ll


def test_per_animal_si_logistic_xG_binned_returns_finite():
    m = _make_mouse(seed=35, noise_amp=0.05, n_easy_per_side=25)
    feats = dre.compute_features(m)
    c = feats["signed_contrast"]
    m.choice = (c > 0).astype(float)
    proto = dre._compute_archetypes(m)
    ex = dre._compute_archetypes_exemplars(m)
    fit = dre._per_animal_si_logistic_xG_binned(
        m, proto, ex, method="prototype", n_bins=4)
    assert np.isfinite(fit["ll_si"])
    assert np.isfinite(fit["ll_stim"])
    assert fit["n_features"] == 4


def test_fit_interrogation_ddm_recovers_planted_slope():
    """Generate choices from a known DDM, recover α."""
    rng = _rng(34)
    n = 500
    c = rng.uniform(-1, 1, n)
    alpha_true, bias_true, sv_true = 2.5, 0.2, 0.3
    T = 2.0
    # Generate choices under the model
    v_i = alpha_true * c + sv_true * rng.standard_normal(n)
    X_T = v_i * T + np.sqrt(T) * rng.standard_normal(n) + bias_true * T
    choice = (X_T > 0).astype(float)
    fit = dre._fit_interrogation_ddm(c, choice, T=T)
    assert np.isfinite(fit["alpha"])
    # Recovery should be within reasonable tolerance
    assert abs(fit["alpha"] - alpha_true) < 0.8, (
        f"recovered α = {fit['alpha']:.3f}, true = {alpha_true}")


def test_fit_interrogation_ddm_handles_insufficient_data():
    fit = dre._fit_interrogation_ddm(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert all(np.isnan(fit[k]) for k in
               ("alpha", "bias", "sigma_v", "lapse_L", "lapse_R"))
