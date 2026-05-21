"""The four bundle-similarity variants from the framework's open question.

A bundle archetype admits several non-equivalent operationalisations
(Similarity Framework, "Bundle representation"):

* **prototype**          -- cosine to the bundle mean trajectory.
* **expected_exemplar**  -- mean cosine over the K stored exemplar trajectories
                            (a kernel-density estimate of the class likelihood).
* **max_exemplar**       -- nearest stored exemplar (Nosofsky-style GCM).
* **vmf**                -- a von Mises-Fisher likelihood, concentration kappa
                            tied to bundle width; SI is then tanh of the
                            log-likelihood ratio.

All four are computed here as alternative *readouts of the same learned
representation*, so they can be compared head-to-head (framework Prediction 2).

Note on vMF: an exact high-dimensional vMF needs a Bessel normalisation of
order n_v1/2 (intractable here without scipy). We use a single per-animal
concentration kappa, which makes the vMF SI = tanh(kappa * (cos_go - cos_nogo)
/ 2) -- a concentration-scaled, saturating prototype. It is the correct
"parametric concentrated" limit; with one kappa it is monotonically related to
the prototype, so it is expected to track (not diverge from) the prototype.
"""

from __future__ import annotations

import numpy as np

from .decision import cosine_trace

VARIANT_NAMES: tuple[str, ...] = (
    "prototype", "expected_exemplar", "max_exemplar", "vmf")


def cosine_to_bundle(r: np.ndarray, bundle: np.ndarray, eps: float) -> np.ndarray:
    """Pointwise cosine of r(t) to every exemplar trajectory in a bundle.

    ``r`` is ``(B, n_steps, n_v1)``, ``bundle`` is ``(K, n_steps, n_v1)``.
    Returns ``(B, n_steps, K)``.
    """
    dot = np.einsum("btn,ktn->btk", r, bundle)
    norm_r = np.linalg.norm(r, axis=-1)            # (B, n_steps)
    norm_b = np.linalg.norm(bundle, axis=-1)       # (K, n_steps)
    denom = norm_r[:, :, None] * norm_b.T[None, :, :] + eps
    return dot / denom


def _si_from_similarities(s_go: np.ndarray, s_nogo: np.ndarray,
                          eps: float) -> np.ndarray:
    """Bounded normalised difference; positive favours Go."""
    return (s_go - s_nogo) / (s_go + s_nogo + eps)


def si_prototype(r: np.ndarray, proto_go: np.ndarray, proto_nogo: np.ndarray,
                 eps: float) -> np.ndarray:
    """Prototype variant: cosine to the bundle-mean trajectory."""
    s_go = (1.0 + cosine_trace(r, proto_go, eps)) / 2.0
    s_nogo = (1.0 + cosine_trace(r, proto_nogo, eps)) / 2.0
    return _si_from_similarities(s_go, s_nogo, eps)


def si_expected_exemplar(r: np.ndarray, bundle_go: np.ndarray,
                         bundle_nogo: np.ndarray, eps: float) -> np.ndarray:
    """Expected-exemplar variant: mean positivity-mapped cosine over the bundle."""
    s_go = ((1.0 + cosine_to_bundle(r, bundle_go, eps)) / 2.0).mean(axis=2)
    s_nogo = ((1.0 + cosine_to_bundle(r, bundle_nogo, eps)) / 2.0).mean(axis=2)
    return _si_from_similarities(s_go, s_nogo, eps)


def si_max_exemplar(r: np.ndarray, bundle_go: np.ndarray,
                    bundle_nogo: np.ndarray, eps: float) -> np.ndarray:
    """Max-exemplar variant: similarity to the single nearest exemplar."""
    s_go = ((1.0 + cosine_to_bundle(r, bundle_go, eps)) / 2.0).max(axis=2)
    s_nogo = ((1.0 + cosine_to_bundle(r, bundle_nogo, eps)) / 2.0).max(axis=2)
    return _si_from_similarities(s_go, s_nogo, eps)


def si_vmf(r: np.ndarray, proto_go: np.ndarray, proto_nogo: np.ndarray,
           kappa: float, eps: float) -> np.ndarray:
    """vMF variant: SI = tanh of the (concentration-weighted) log-odds."""
    cos_go = cosine_trace(r, proto_go, eps)
    cos_nogo = cosine_trace(r, proto_nogo, eps)
    return np.tanh(0.5 * kappa * (cos_go - cos_nogo))


def estimate_kappa(bundle_go: np.ndarray, bundle_nogo: np.ndarray) -> float:
    """Per-animal vMF concentration from the exemplar bundles.

    For each class the exemplar trajectories are compared (pointwise cosine)
    to the bundle-mean direction; kappa is the reciprocal of the mean angular
    width, pooled over both classes.
    """
    widths = []
    for bundle in (bundle_go, bundle_nogo):
        mean_traj = bundle.mean(axis=0)                     # (n_steps, n_v1)
        unit = bundle / (np.linalg.norm(bundle, axis=-1, keepdims=True) + 1e-9)
        unit_mean = mean_traj / (np.linalg.norm(mean_traj, axis=-1,
                                                keepdims=True) + 1e-9)
        cos = np.einsum("ktn,tn->kt", unit, unit_mean)
        widths.append(float(np.mean(1.0 - np.clip(cos, -1.0, 1.0))))
    return 1.0 / (float(np.mean(widths)) + 1e-9)


def all_variant_si(r: np.ndarray, proto_go: np.ndarray, proto_nogo: np.ndarray,
                   bundle_go: np.ndarray, bundle_nogo: np.ndarray,
                   kappa: float, eps: float) -> dict[str, np.ndarray]:
    """Compute SI(t) under all four variants for a batch of trajectories."""
    return {
        "prototype": si_prototype(r, proto_go, proto_nogo, eps),
        "expected_exemplar": si_expected_exemplar(r, bundle_go, bundle_nogo, eps),
        "max_exemplar": si_max_exemplar(r, bundle_go, bundle_nogo, eps),
        "vmf": si_vmf(r, proto_go, proto_nogo, kappa, eps),
    }
