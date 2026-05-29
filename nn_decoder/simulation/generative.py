# -*- coding: utf-8 -*-
"""Forward generative simulation of PPC and SBC neural populations.

This is the *generative* counterpart to the project's readouts in
``utils.generate_PPC_targets`` / ``generate_SBC_targets`` (which go the
other way: activity -> distribution). Here we start from **known latents**
— a true stimulus, a likelihood, a prior, a posterior, a decision — and
synthesise a population of neurons whose activity encodes those latents
under one of the two competing hypotheses, so we can then fit every
decoder and check which recovers the latents (model recovery).

Two canonical textbook generative models (chosen for this study):

PPC — linear probabilistic population code (Ma, Beck, Latham & Pouget,
    *Nat. Neurosci.* 2006). Neurons have bell-shaped tuning ``f_n(theta)``
    and emit independent Poisson spike counts ``r_{n,t} ~ Poisson(g f_n(s*))``
    with the SAME mean in every time bin. For Poisson noise the
    log-likelihood over theta is linear in the summed counts, so the
    population's *time-summed* activity is a sufficient statistic and the
    width of the encoded likelihood scales as ``1/sqrt(total count)`` — i.e.
    **gain encodes certainty**. Temporal order carries no information, so a
    mean-then-decode (PPC) readout is optimal.

SBC — neural sampling (Hoyer & Hyvärinen 2003; Fiser et al. 2010). Each
    time bin carries one *sample* ``theta_t ~ Q*(theta)`` drawn from the
    posterior, rendered as a sharp tuning bump ``r_{n,t} ~ Poisson(g f_n(theta_t))``.
    Across bins the samples trace out the posterior, so **uncertainty is
    the spread of samples over time** and a per-bin-decode-then-average
    (SBC) readout is optimal; time-averaging the activity collapses the
    spread and destroys the uncertainty signal.

The two schemes therefore encode uncertainty by genuinely different
mechanisms — spike count (PPC) vs sample spread (SBC). That difference is
the hypothesis under test, so the resulting total-spike-count difference
between schemes is intrinsic, not a confound to be removed.

All distributions live on the integer orientation grid 0..90 deg (91
bins), matching the rest of the decoder pipeline. Activity arrays use the
repo's neurons-first contract ``(n_neurons, n_trials, T)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


S_GRID = np.arange(0, 91, 1)          # 91-bin orientation grid (deg)
GO_CUTOFF_DEG = 45                    # theta < 45 -> Go (prior mode at 0), else No-Go


# ----------------------------------------------------------------------
# Tuning curves
# ----------------------------------------------------------------------

def make_tuning_curves(n_neurons, s_grid=S_GRID, tuning_width=12.0,
                       baseline=0.5, peak=1.0, jitter=0.0, pad=None, seed=0):
    """Bell-shaped (Gaussian) tuning curves tiling the orientation grid.

    Returns ``f`` of shape ``(len(s_grid), n_neurons)`` — the expected
    spike count per bin at unit gain — and the per-neuron preferred
    orientations ``pref`` of shape ``(n_neurons,)``.

    Preferred orientations are spread evenly across a range that is
    **padded beyond the grid** by ``pad`` degrees (default ``2.5 *
    tuning_width``). The padding keeps the summed population rate
    ``sum_n f_n(theta)`` flat across the interior of ``[0, 90]``; without
    it the population thins out at the edges and the ``- sum_n f_n(theta)``
    term of the Poisson log-likelihood biases the ML readout toward the
    grid boundaries. A positive ``baseline`` keeps every rate strictly
    above zero so ``r log f - f`` is finite everywhere.
    """
    rng = np.random.RandomState(seed)
    s_grid = np.asarray(s_grid, dtype=float)
    if pad is None:
        pad = 2.5 * tuning_width
    pref = np.linspace(s_grid.min() - pad, s_grid.max() + pad, n_neurons)
    if jitter > 0:
        pref = pref + rng.normal(0, jitter, n_neurons)
    # (n_grid, n_neurons) Gaussian bumps + baseline.
    d = s_grid[:, None] - pref[None, :]
    f = baseline + peak * np.exp(-0.5 * (d / tuning_width) ** 2)
    return np.clip(f, 1e-6, None), pref


# ----------------------------------------------------------------------
# Latents: likelihood, prior, posterior, decision
# ----------------------------------------------------------------------

def make_prior(s_grid=S_GRID, prior_type='bimodal', prior_sigma=15.0):
    """Task prior over orientation. Bimodal = mixture at the Go (0 deg) and
    No-Go (90 deg) modes; flat = uniform. Mirrors ``utils.get_prior``."""
    s_grid = np.asarray(s_grid, dtype=float)
    if prior_type == 'bimodal':
        p = (np.exp(-0.5 * (s_grid - 0) ** 2 / prior_sigma ** 2) +
             np.exp(-0.5 * (s_grid - 90) ** 2 / prior_sigma ** 2))
    else:
        p = np.ones_like(s_grid)
    return p / p.sum()


def _gaussian_on_grid(center, sigma, s_grid=S_GRID):
    g = np.exp(-0.5 * ((s_grid - center) / sigma) ** 2)
    return g / g.sum()


@dataclass
class Latents:
    """Per-trial ground-truth latents the decoders must recover.

    s_true     : (n_trials,)        true stimulus orientation (deg)
    sigma      : (n_trials,)        likelihood SD (deg) — the uncertainty
    level      : (n_trials,)        integer index into uncertainty_levels
    likelihood : (n_trials, 91)     L*(theta) — Gaussian bump at s_true
    prior      : (91,)              p(theta)
    posterior  : (n_trials, 91)     Q* propto L* . p
    decision   : (n_trials, 2)      [P(Go), P(No-Go)] from Q*
    """
    s_true: np.ndarray
    sigma: np.ndarray
    level: np.ndarray
    likelihood: np.ndarray
    prior: np.ndarray
    posterior: np.ndarray
    decision: np.ndarray


def make_latents(n_trials, uncertainty_levels=(4.0, 8.0, 16.0),
                 s_grid=S_GRID, prior_type='bimodal', s_range=(10, 80),
                 go_cutoff=GO_CUTOFF_DEG, seed=0):
    """Draw per-trial latents.

    Each trial gets a uncertainty *level* (a likelihood SD drawn uniformly
    from ``uncertainty_levels``) and a true stimulus ``s_true`` drawn
    uniformly from ``s_range`` (kept off the grid edges so the Gaussian
    likelihood bump is not truncated). The likelihood is a Gaussian bump at
    ``s_true`` with that SD; the posterior is ``L* . prior`` renormalised;
    the decision posterior integrates ``Q*`` over the Go / No-Go regions.
    """
    rng = np.random.RandomState(seed)
    levels = np.asarray(uncertainty_levels, dtype=float)
    level_idx = rng.randint(0, len(levels), size=n_trials)
    sigma = levels[level_idx]
    s_true = rng.uniform(s_range[0], s_range[1], size=n_trials)

    prior = make_prior(s_grid, prior_type)
    L = np.stack([_gaussian_on_grid(s_true[i], sigma[i], s_grid)
                  for i in range(n_trials)])
    P_unnorm = L * prior[None, :]
    Q = P_unnorm / P_unnorm.sum(axis=1, keepdims=True)

    go_mask = np.asarray(s_grid) < go_cutoff
    p_go = Q[:, go_mask].sum(axis=1)
    decision = np.column_stack([p_go, 1.0 - p_go])

    return Latents(s_true=s_true, sigma=sigma, level=level_idx,
                   likelihood=L, prior=prior, posterior=Q, decision=decision)


# ----------------------------------------------------------------------
# Gain calibration (PPC): map a desired likelihood width to a Poisson gain
# ----------------------------------------------------------------------

def population_fisher(tuning, s_grid=S_GRID):
    """Mean (over the grid) Poisson population Fisher information per unit
    gain per time bin: ``J(theta) = sum_n f'_n(theta)^2 / f_n(theta)``,
    averaged over theta. Used to convert a target posterior SD into the
    Poisson gain that yields it (posterior variance ~ 1 / total Fisher)."""
    fp = np.gradient(tuning, np.asarray(s_grid, dtype=float), axis=0)
    J = np.sum(fp ** 2 / tuning, axis=1)
    return float(np.mean(J))


def calibrate_gain(sigma, tuning, T, s_grid=S_GRID, gain_floor=1e-3):
    """Poisson gain so the linear-PPC likelihood has SD ~ ``sigma`` after
    pooling ``T`` time bins. From the Cramer-Rao bound the posterior
    variance is ~ ``1 / (gain * T * J)`` with ``J`` the population Fisher
    info, so ``gain = 1 / (sigma^2 * T * J)``. Higher certainty (smaller
    sigma) -> higher gain -> more spikes, the canonical PPC relationship.
    Accepts scalar or array ``sigma``; returns the matching shape."""
    J = population_fisher(tuning, s_grid)
    gain = 1.0 / (np.asarray(sigma, dtype=float) ** 2 * T * J)
    return np.clip(gain, gain_floor, None)


# ----------------------------------------------------------------------
# Population simulators
# ----------------------------------------------------------------------

def simulate_ppc(latents, tuning, T, s_grid=S_GRID, seed=0):
    """Linear-PPC population: every time bin is an independent Poisson draw
    with the SAME per-neuron mean ``gain_i * f_n(s_true_i)``, where the
    per-trial gain is calibrated so the encoded likelihood width matches
    the trial's ``sigma``. Returns activity ``(n_neurons, n_trials, T)``.

    Because every bin shares the mean, the time-summed count is a
    sufficient statistic — a mean-then-decode readout loses nothing, while
    a per-bin sampling readout sees only Poisson noise around a fixed bump.
    """
    rng = np.random.RandomState(seed)
    s_grid = np.asarray(s_grid)
    n_neurons = tuning.shape[1]
    n_trials = len(latents.s_true)
    gain = calibrate_gain(latents.sigma, tuning, T, s_grid)

    # Per-trial mean rate at the true stimulus (nearest grid bin).
    idx = np.clip(np.rint(latents.s_true).astype(int),
                  s_grid.min(), s_grid.max())
    f_at_s = tuning[idx, :]                          # (n_trials, n_neurons)
    mean_rate = gain[:, None] * f_at_s               # (n_trials, n_neurons)

    # Poisson counts, identical mean across the T bins.
    lam = np.broadcast_to(mean_rate[:, None, :], (n_trials, T, n_neurons))
    counts = rng.poisson(lam).astype(float)          # (n_trials, T, n_neurons)
    return np.transpose(counts, (2, 0, 1))           # (n_neurons, n_trials, T)


def simulate_sbc(latents, tuning, T, sample_gain=12.0, sample_from='posterior',
                 s_grid=S_GRID, seed=0):
    """Neural-sampling population: each time bin draws one sample
    ``theta_t`` from the trial's posterior (or likelihood) and renders a
    sharp Poisson tuning bump ``Poisson(sample_gain * f_n(theta_t))``.
    Across the T bins the samples are distributed as the sampled
    distribution, so its spread *is* the encoded uncertainty. Returns
    activity ``(n_neurons, n_trials, T)``.

    ``sample_gain`` is held fixed (sharp, identifiable bumps) so that
    per-bin decoding recovers ``theta_t`` reliably; uncertainty is carried
    by the spread of samples, not by spike count — the opposite of PPC.

    ``sample_from`` selects the distribution sampled: 'posterior' (the
    neural-sampling hypothesis samples Q*) or 'likelihood' (sample L*).
    Note the asymmetry this induces: when sampling the posterior, the
    likelihood is not separately represented, so recovering L* from SBC
    activity is intrinsically hard — a genuine, interpretable feature of
    the sampling hypothesis, not an artefact.
    """
    rng = np.random.RandomState(seed)
    s_grid = np.asarray(s_grid)
    n_neurons = tuning.shape[1]
    n_trials = len(latents.s_true)
    dist = latents.posterior if sample_from == 'posterior' else latents.likelihood

    activity = np.empty((n_neurons, n_trials, T))
    grid_idx = np.arange(len(s_grid))
    for i in range(n_trials):
        sample_idx = rng.choice(grid_idx, size=T, p=dist[i])   # theta_t bins
        lam = sample_gain * tuning[sample_idx, :]              # (T, n_neurons)
        activity[:, i, :] = rng.poisson(lam).T                 # (n_neurons, T)
    return activity


def simulate_population(scheme, latents, tuning, T, *, seed=0, **kwargs):
    """Dispatch to the PPC or SBC simulator by name ('ppc' | 'sbc')."""
    if scheme == 'ppc':
        return simulate_ppc(latents, tuning, T, seed=seed,
                            s_grid=kwargs.get('s_grid', S_GRID))
    if scheme == 'sbc':
        return simulate_sbc(latents, tuning, T, seed=seed,
                            sample_gain=kwargs.get('sample_gain', 12.0),
                            sample_from=kwargs.get('sample_from', 'posterior'),
                            s_grid=kwargs.get('s_grid', S_GRID))
    raise ValueError(f"Unknown scheme {scheme!r}; expected 'ppc' or 'sbc'")
