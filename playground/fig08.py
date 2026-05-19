"""Figure 8 — κ vs trial-mean-SI spread across animals (Prediction 4).

Renamed from the original "κ vs DDM $s_v$" framing. What we compute and
plot here is a *neural summary statistic*: the variance, across trials at
the strongest evidence level, of the trial-averaged SI under the
expected-exemplar method. This is NOT a DDM-fit parameter; a proper test
of bundle-width-versus-DDM-$s_v$ would require fitting a DDM to behaviour
(deferred ``similarity_fit.py``).

Left panel: 8 synthetic mice with planted κ values, recover the
trial-mean-SI spread via the expected-exemplar SI; the framework predicts
a clean negative correlation (tight bundle → consistent SI across max-
contrast trials → low spread). Right panel: the same spreads shuffled so
κ↔spread are decoupled — the failure mode where the bundle-width-predicts-
neural-SI-spread mechanism does not hold.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import framework_common as fc


def _synth_animal(kappa_target, n_trials=200, n_per_bundle=20, seed=0):
    """One synthetic 'animal' with a controlled bundle width.

    kappa_target maps inversely to the angular spread of the exemplar
    bundle. We construct exemplars with the appropriate spread, then
    simulate test trials and compute trial-mean SI.
    """
    rng = np.random.default_rng(seed)
    # Convert target κ back to an angular spread (rough inverse via
    # mean-resultant-length ≈ 1/sqrt(1 + 2/κ) in 2-D vMF).
    angular_spread = 1.0 / np.sqrt(max(kappa_target, 0.5))
    # Generate bundles per side
    bundles = {}
    for side, end in [("L", fc.END_L), ("R", fc.END_R)]:
        mean = fc.START + fc.RAMP[:, None] * (end - fc.START)
        b = np.empty((n_per_bundle, fc.N, 2))
        for k in range(n_per_bundle):
            theta = angular_spread * rng.standard_normal()
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            b[k] = (mean - fc.START) @ rot.T + fc.START
            b[k] += fc.ou_traj(fc.N, sigma=0.04, seed=seed * 9000 + k)
        bundles[side] = b

    # Generate "test" trials randomly drawn from either side's distribution
    trial_mean_si = np.empty(n_trials)
    for t_idx in range(n_trials):
        side = "R" if rng.random() < 0.5 else "L"
        theta = angular_spread * rng.standard_normal()
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        end = fc.END_R if side == "R" else fc.END_L
        mean_traj = fc.START + fc.RAMP[:, None] * (end - fc.START)
        r_t = (mean_traj - fc.START) @ rot.T + fc.START
        r_t += fc.ou_traj(fc.N, sigma=0.06, seed=seed * 7000 + t_idx)
        # Expected-exemplar SI
        sL_k = np.stack([fc.cosine_sim(r_t, bundles["L"][k])
                         for k in range(n_per_bundle)])
        sR_k = np.stack([fc.cosine_sim(r_t, bundles["R"][k])
                         for k in range(n_per_bundle)])
        sL = sL_k.mean(axis=0); sR = sR_k.mean(axis=0)
        si = ((1 + sR) / 2 - (1 + sL) / 2) / ((1 + sR) / 2 + (1 + sL) / 2 + 1e-9)
        trial_mean_si[t_idx] = si.mean()

    # Recover κ from the bundles
    K = n_per_bundle
    p_dim = 2
    kappas = []
    for side in ("L", "R"):
        Ek_hat = bundles[side] / (np.linalg.norm(bundles[side], axis=2,
                                                 keepdims=True) + 1e-9)
        resultant = Ek_hat.sum(axis=0)
        R_len = np.linalg.norm(resultant, axis=1)
        R_bar = R_len / K
        R_bar = np.clip(R_bar, 0.0, 1 - 1e-6)
        k_t = R_bar * (p_dim - R_bar ** 2) / (1.0 - R_bar ** 2)
        kappas.append(k_t.mean())
    kappa_recovered = 0.5 * (kappas[0] + kappas[1])

    si_spread = float(np.var(trial_mean_si))
    return kappa_recovered, si_spread


def main():
    target_kappas = [2.0, 4.0, 8.0, 15.0, 25.0, 40.0, 70.0, 110.0]
    n_animals = len(target_kappas)
    rec = np.array([_synth_animal(kt, seed=11 * i) for i, kt in enumerate(target_kappas)])
    kappas, spreads = rec[:, 0], rec[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    fig.suptitle("Figure 8 — Bundle κ vs trial-mean-SI spread "
                 "(neural summary statistic; not DDM $s_v$)",
                 fontsize=11, y=0.98)

    # Framework prediction
    ax = axes[0, 0]
    ax.scatter(kappas, spreads, s=80, c="steelblue", edgecolors="black",
               zorder=3)
    for i, (k, s) in enumerate(zip(kappas, spreads)):
        ax.annotate(f"m{i+1}", (k, s), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    r, p = pearsonr(kappas, spreads)
    ax.set_title(f"Framework prediction\nPearson r = {r:.2f}, p = {p:.3f}",
                 fontsize=10)
    ax.set_xlabel("recovered bundle κ", fontsize=9)
    ax.set_ylabel(r"trial-mean-SI spread $\mathrm{Var}_{\mathrm{trial}}[\mathbb{E}_t SI]$",
                  fontsize=9)
    ax.tick_params(labelsize=8)

    # Failure mode: κ and spread decoupled (shuffle the pairs).
    rng = np.random.default_rng(99)
    perm = rng.permutation(n_animals)
    ax = axes[0, 1]
    ax.scatter(kappas, spreads[perm], s=80, c="darkgray", edgecolors="black",
               zorder=3)
    for i, (k, s) in enumerate(zip(kappas, spreads[perm])):
        ax.annotate(f"m{i+1}", (k, s), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    r_null, p_null = pearsonr(kappas, spreads[perm])
    ax.set_title(f"Failure mode (κ↔spread decoupled)\n"
                 f"Pearson r = {r_null:.2f}, p = {p_null:.3f}", fontsize=10)
    ax.set_xlabel("recovered bundle κ", fontsize=9)
    ax.set_ylabel(r"trial-mean-SI spread", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{fc.OUTDIR}/08_kappa_vs_si_spread.png"
    fig.savefig(out)
    plt.close(fig)
    print("Figure 8 saved.")


if __name__ == "__main__":
    main()
