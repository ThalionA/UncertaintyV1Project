"""
Figure 16 — spatial clash test: gain manipulation should NOT shift κ (Prediction 10).

spatial says population gain encodes posterior precision — manipulate the
gain (contrast, attention) and the represented precision shifts. This
framework says cosine throws away magnitude, so the bundle's angular
concentration κ is invariant to gain unless the manipulation also recruits
different exemplars. The two predictions are dissociable: simulate
bundles under low vs high gain, fit κ, and check whether it moves.

Layout: two columns:
  Left  — framework prediction: κ unchanged across gain levels.
  Right — spatial prediction: κ scales with gain (proportional to gain^2 in
          the simplest spatial).

Note: this test requires an experimental manipulation not in the current
UncertaintyV1 dataset, so the panel is theoretical-only — a "what to
look for" reference for future experiments.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

import framework_common as fc
from framework_common import OUTDIR, RAMP, START, ou_traj, C_L, C_R


def _build_bundle(end, n=20, seed=0, gain=1.0, angular_spread=0.10):
    """Build a bundle of n exemplars with given gain (magnitude scaling)
    and angular spread (intrinsic bundle width)."""
    rng = np.random.default_rng(seed)
    mean = START + RAMP[:, None] * (end - START)
    bundle = np.empty((n, fc.N, 2))
    for k in range(n):
        theta = angular_spread * rng.standard_normal()
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = (mean - START) @ rot.T + START
        # Apply gain — magnitude only, leaves direction unchanged
        bundle[k] = START + gain * (rotated - START)
        bundle[k] += 0.05 * gain * rng.standard_normal((fc.N, 2))
    return bundle


def _kappa(bundle, p_dim=2):
    """Banerjee κ averaged over time."""
    K = bundle.shape[0]
    norm = np.linalg.norm(bundle, axis=2, keepdims=True) + 1e-9
    Ek_hat = bundle / norm
    resultant = Ek_hat.sum(axis=0)
    R_len = np.linalg.norm(resultant, axis=1)
    R_bar = R_len / K
    R_bar = np.clip(R_bar, 0.0, 1.0 - 1e-6)
    kappa = R_bar * (p_dim - R_bar ** 2) / (1.0 - R_bar ** 2)
    return float(kappa.mean())


gains = np.linspace(0.5, 2.0, 6)

# Framework: κ is angular-only → invariant to gain (ideally).
# spatial: κ ∝ gain^2 (signal-to-noise scaling under Poisson-like models).
kappa_framework_L = [_kappa(_build_bundle(fc.END_L, gain=g, seed=11))
                    for g in gains]
kappa_framework_R = [_kappa(_build_bundle(fc.END_R, gain=g, seed=12))
                    for g in gains]

# spatial predicted κ: scales as gain^2 (illustrative; the exact scaling
# depends on noise model, but the qualitative point is positive monotonic).
kappa_ppc_L = [kf * (g ** 2) for kf, g in zip(kappa_framework_L, gains)]
kappa_ppc_R = [kf * (g ** 2) for kf, g in zip(kappa_framework_R, gains)]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), squeeze=False)
fig.suptitle("Figure 16 — Spatial clash test: does gain manipulation shift κ?",
             fontsize=11)

# Framework
ax = axes[0, 0]
ax.plot(gains, kappa_framework_L, "o-", color=C_L, lw=1.5,
        markersize=8, label="L bundle")
ax.plot(gains, kappa_framework_R, "s-", color=C_R, lw=1.5,
        markersize=8, label="R bundle")
ax.set_xlabel("population gain (relative to baseline)", fontsize=9)
ax.set_ylabel("recovered κ", fontsize=9)
ax.set_title("Framework prediction:  κ ⊥ gain\n"
             "(cosine throws away magnitude → only direction matters)",
             fontsize=10)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# spatial
ax = axes[0, 1]
ax.plot(gains, kappa_ppc_L, "o-", color=C_L, lw=1.5,
        markersize=8, label="L bundle")
ax.plot(gains, kappa_ppc_R, "s-", color=C_R, lw=1.5,
        markersize=8, label="R bundle")
ax.set_xlabel("population gain (relative to baseline)", fontsize=9)
ax.set_ylabel("recovered κ", fontsize=9)
ax.set_title("Spatial prediction:  κ ∝ gain²\n"
             "(gain encodes posterior precision under spatial)",
             fontsize=10)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

fig.tight_layout(rect=(0, 0, 1, 0.94))
out = f"{OUTDIR}/16_ppc_clash_test.png"
fig.savefig(out)
plt.close(fig)
print("Figure 16 saved.")
