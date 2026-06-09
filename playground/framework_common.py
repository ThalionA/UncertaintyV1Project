"""Common setup shared across all 6 framework figures.

Output directory: set the SIMFRAME_OUTDIR environment variable to override.
Default writes to ./figures/ next to this file.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("SIMFRAME_OUTDIR",
                       os.path.join(_THIS_DIR, "figures"))
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

# ----- palette -----
C_L = "#d62728"
C_R = "#1f77b4"
C_L_LIGHT = "#f4a8a8"
C_R_LIGHT = "#a8c9e8"
C_EASY = "#2ca02c"
C_AMB = "#9467bd"
C_R_RESP = "#1a1a1a"     # response trajectory r(t)
C_GRAY = "#888888"

# ----- simulation parameters -----
# Coarse time grid: 20 timesteps of 100 ms over a 2 s trial. Coarsening
# was an explicit visual choice — dense (200-step) trajectories
# overwhelmed several panels with noise; the 20-step view keeps the
# story-telling clean while still resolving the within-trial dynamics
# the framework relies on.
DT = 0.1
T_TRIAL = 2.0
T = np.arange(0.0, T_TRIAL, DT)
N = len(T)

#
# Two prototype regimes are exposed so figures can show how the framework's
# distinctive predictions depend on stimulus discriminability:
#
#   REALISTIC (close prototypes, default ``END_L`` / ``END_R``) — closer to
#     real V1, where both stimulus classes share most of their response and
#     discriminative information lives in a small angular subspace. This is
#     the regime where the framework's bundle-width effects, variants
#     divergence, and temporal clash predictions become diagnostic.
#
#   EASY (well-separated prototypes, ``END_L_EASY`` / ``END_R_EASY``) —
#     the original polar geometry. Discrimination is trivial and the
#     framework's distinctive predictions collapse: any reasonable
#     similarity measure works, bundle width barely matters, variants
#     overlap regardless of method.
#
# Figures that show one regime tell only half the story; figures that show
# both can demonstrate the predicted interaction with stimulus difficulty.
START = np.array([0.5, 0.5])

# Realistic / close — default for fig01–fig06 panels labelled "hard"
END_L = np.array([2.3, 2.6])   # ~48° from origin
END_R = np.array([2.9, 1.4])   # ~26° from origin

# Easy / well-separated — for fig01–fig06 panels labelled "easy"
END_L_EASY = np.array([-2.2, 1.6])
END_R_EASY = np.array([2.4, -0.3])

# Smooth ramp baseline -> archetype endpoint. Tuned to saturate around
# t ≈ 1 s under the 2 s trial — so trials still show clear dynamics.
RAMP = 1.0 - np.exp(-T / 0.35)
MEAN_L = START + RAMP[:, None] * (END_L - START)
MEAN_R = START + RAMP[:, None] * (END_R - START)


def ou_traj(n_steps, sigma=0.22, tau=0.30, dim=2, seed=None):
    """OU process noise. Default ``tau`` is 3 × ``DT`` so a single time
    step preserves about two-thirds of the previous state — gives moderate
    autocorrelation under the coarse 20-step grid."""
    rng = np.random.default_rng(seed)
    eta = np.zeros((n_steps, dim))
    a = DT / tau
    s = sigma * np.sqrt(2 * DT / tau)
    for i in range(1, n_steps):
        eta[i] = (1 - a) * eta[i-1] + s * rng.standard_normal(dim)
    return eta


K = 14
TRAJS_L = np.stack([MEAN_L + ou_traj(N, seed=100 + k) for k in range(K)])
TRAJS_R = np.stack([MEAN_R + ou_traj(N, seed=200 + k) for k in range(K)])


def cosine_sim(r, a):
    """Pointwise cosine sim between two (T, D) trajectories."""
    dot = np.einsum("ij,ij->i", r, a)
    nr = np.linalg.norm(r, axis=1) + 1e-9
    na = np.linalg.norm(a, axis=1) + 1e-9
    return dot / (nr * na)


def cosine_sim_3d(r, a):
    """Cosine sim for 3D arrays."""
    return cosine_sim(r, a)


def normalize(traj):
    """Pointwise normalisation to unit norm."""
    nrm = np.linalg.norm(traj, axis=1, keepdims=True) + 1e-9
    return traj / nrm


def angles_deg(traj):
    return np.degrees(np.arctan2(traj[:, 1], traj[:, 0]))


# ----- bounded SI (Wiener form) -----
def si_normalized(r, mean_L, mean_R):
    """Bounded SI in [-1, +1] using positivity-mapped cosine."""
    sL = (1 + cosine_sim(r, mean_L)) / 2.0
    sR = (1 + cosine_sim(r, mean_R)) / 2.0
    return (sR - sL) / (sR + sL + 1e-9)


# ----- shorter arc helper -----
def shorter_arc(theta1, theta2):
    """Return ordered (theta_start, theta_end) for the shorter arc CCW."""
    diff_ccw = (theta2 - theta1) % 360
    return (theta1, theta2) if diff_ccw <= 180 else (theta2, theta1)


if __name__ == "__main__":
    print(f"OUTDIR = {OUTDIR}")
    print(f"K = {K}, N timesteps = {N}, T_trial = {T_TRIAL}s")
    print(f"TRAJS_L shape = {TRAJS_L.shape}")
