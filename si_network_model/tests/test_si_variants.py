"""Tests for si_variants.py -- the four bundle-similarity readouts."""

from __future__ import annotations

import numpy as np

from si_network_model import si_variants
from si_network_model.config import Config
from si_network_model.decision import si_trace


def _setup(cfg: Config, rng: np.random.Generator):
    t, n = cfg.n_steps, cfg.n_v1
    r = np.abs(rng.normal(size=(5, t, n)))
    proto_go = np.abs(rng.normal(size=(t, n))) + 0.1
    proto_nogo = np.abs(rng.normal(size=(t, n))) + 0.1
    return r, proto_go, proto_nogo


def test_cosine_to_bundle_shape_and_bounds():
    cfg = Config(n_v1=30)
    rng = np.random.default_rng(0)
    r = np.abs(rng.normal(size=(5, cfg.n_steps, cfg.n_v1)))
    bundle = np.abs(rng.normal(size=(8, cfg.n_steps, cfg.n_v1)))
    cos = si_variants.cosine_to_bundle(r, bundle, 1e-6)
    assert cos.shape == (5, cfg.n_steps, 8)
    assert np.all(cos >= -1.0 - 1e-6) and np.all(cos <= 1.0 + 1e-6)


def test_all_variants_shapes_and_bounds():
    cfg = Config(n_v1=30)
    rng = np.random.default_rng(1)
    r, pg, pn = _setup(cfg, rng)
    k = cfg.n_exemplars
    bg = np.abs(rng.normal(size=(k, cfg.n_steps, cfg.n_v1))) + 0.1
    bn = np.abs(rng.normal(size=(k, cfg.n_steps, cfg.n_v1))) + 0.1
    kappa = si_variants.estimate_kappa(bg, bn)
    assert kappa > 0.0 and np.isfinite(kappa)
    sis = si_variants.all_variant_si(r, pg, pn, bg, bn, kappa, 1e-6)
    assert set(sis) == set(si_variants.VARIANT_NAMES)
    for name, si in sis.items():
        assert si.shape == (5, cfg.n_steps), name
        assert np.all(si >= -1.0 - 1e-6) and np.all(si <= 1.0 + 1e-6), name


def test_prototype_variant_matches_decision_si_trace():
    cfg = Config(n_v1=30)
    r, pg, pn = _setup(cfg, np.random.default_rng(2))
    assert np.allclose(si_variants.si_prototype(r, pg, pn, cfg.si_eps),
                       si_trace(r, pg, pn, cfg.si_eps))


def test_variants_collapse_for_a_tight_bundle():
    """A bundle of K identical copies of the prototype -> exemplar variants
    reduce exactly to the prototype variant."""
    cfg = Config(n_v1=30)
    r, pg, pn = _setup(cfg, np.random.default_rng(3))
    k = cfg.n_exemplars
    bg, bn = np.tile(pg, (k, 1, 1)), np.tile(pn, (k, 1, 1))
    proto = si_variants.si_prototype(r, pg, pn, cfg.si_eps)
    assert np.allclose(proto, si_variants.si_expected_exemplar(r, bg, bn, cfg.si_eps))
    assert np.allclose(proto, si_variants.si_max_exemplar(r, bg, bn, cfg.si_eps))
