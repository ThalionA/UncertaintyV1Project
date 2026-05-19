"""Figure 14 — Stimulus-locked vs warped alignment.

Prediction 13: stimulus-locked alignment is sufficient — bundle width
should not be dominated by latency jitter. Framework prediction:
recovered κ is similar under stimulus-locked vs (idealised) lick-locked
alignment. Failure mode: stimulus-locked alignment severely inflates the
measured bundle width because the discriminative trajectory is jittered
in time across trials, and warping recovers a much tighter bundle.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import framework_common as fc


def _gen_bundle_with_jitter(jitter_amp, n_trials=30, seed=0):
    """Generate trial trajectories with latency jitter; return (stim_locked,
    warped) bundles."""
    rng = np.random.default_rng(seed)
    n = fc.N
    raw = np.empty((n_trials, n, 2))
    aligned = np.empty((n_trials, n, 2))
    for k in range(n_trials):
        # Shift the start of the ramp by a random latency.
        latency_idx = int(jitter_amp * n * rng.standard_normal())
        shifted_ramp = np.zeros(n)
        # Build a ramp that starts at latency_idx instead of 0
        start = max(0, latency_idx)
        length = n - start
        ramp_segment = 1.0 - np.exp(-np.arange(length) * fc.DT / 0.18)
        shifted_ramp[start:start + length] = ramp_segment
        end = fc.END_R
        mean = fc.START[None, :] + shifted_ramp[:, None] * (end - fc.START)[None, :]
        noise = fc.ou_traj(n, sigma=0.04, seed=seed * 7919 + k)
        raw[k] = mean + noise
        # Warped: shift back by latency_idx (idealised perfect warping).
        warped = np.roll(raw[k], -latency_idx, axis=0)
        aligned[k] = warped
    return raw, aligned


def _bundle_kappa_per_t(traj_set, p_dim=2):
    """Compute Banerjee κ per timestep across the K trial axis."""
    K = traj_set.shape[0]
    norm = np.linalg.norm(traj_set, axis=2, keepdims=True) + 1e-9
    Ek_hat = traj_set / norm
    resultant = Ek_hat.sum(axis=0)
    R_len = np.linalg.norm(resultant, axis=1)
    R_bar = R_len / K
    R_bar = np.clip(R_bar, 0.0, 1.0 - 1e-6)
    return R_bar * (p_dim - R_bar ** 2) / (1.0 - R_bar ** 2)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Figure 14 — Stimulus-locked vs warped: does alignment matter?",
                 fontsize=11)

    # Framework: alignment doesn't change κ very much because latency jitter
    # is small relative to bundle's intrinsic spread.
    ax = axes[0, 0]
    raw_low, aligned_low = _gen_bundle_with_jitter(jitter_amp=0.05, seed=11)
    kappa_raw = _bundle_kappa_per_t(raw_low)
    kappa_aligned = _bundle_kappa_per_t(aligned_low)
    ax.plot(fc.T, kappa_raw, color="#1f77b4", lw=1.6, label="stimulus-locked")
    ax.plot(fc.T, kappa_aligned, color="#2ca02c", lw=1.6, label="idealised warped")
    ax.set_xlabel("time within trial (s)", fontsize=9)
    ax.set_ylabel("recovered κ", fontsize=9)
    ax.set_title("Framework prediction (low jitter)\n"
                 "stimulus-locked and warped recoveries agree",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # Failure: large jitter makes stimulus-locking severely inflate width.
    ax = axes[0, 1]
    raw_high, aligned_high = _gen_bundle_with_jitter(jitter_amp=0.30, seed=11)
    kappa_raw_h = _bundle_kappa_per_t(raw_high)
    kappa_aligned_h = _bundle_kappa_per_t(aligned_high)
    ax.plot(fc.T, kappa_raw_h, color="#1f77b4", lw=1.6, label="stimulus-locked")
    ax.plot(fc.T, kappa_aligned_h, color="#2ca02c", lw=1.6, label="idealised warped")
    ax.set_xlabel("time within trial (s)", fontsize=9)
    ax.set_ylabel("recovered κ", fontsize=9)
    ax.set_title("Failure mode (high jitter)\n"
                 "stimulus-locked κ is inflated; warping recovers true width",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"{fc.OUTDIR}/14_alignment_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 14 saved.")


if __name__ == "__main__":
    main()
