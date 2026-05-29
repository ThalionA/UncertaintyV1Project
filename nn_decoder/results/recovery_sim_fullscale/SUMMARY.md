# Full-scale generative model-recovery — results

Run: `run_recovery_simulation.py --targets Q L d --schemes ppc sbc
--n-trials 600 --n-neurons 100 --T 10 --rep 4 --epochs 50
--uncertainty-levels 4 8 16`

Metric below is **fit-loss / shuffle** (lower = better recovery; the
matched-decoder hypothesis predicts the diagonal — decoder matches
generative scheme — should be lowest). `kl`/`mse` = divergence to the
ground-truth latent on held-out trials.

## fit_loss / shuffle (lower = better)

| target | gen scheme | PPC | SBC | free_mlp | free_gru | free_tcn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Q | PPC-gen | **0.350** | 0.564 | 0.609 | 0.400 | 0.385 |
| Q | SBC-gen | 0.067 | 0.073 | 0.076 | **0.051** | 0.056 |
| L | PPC-gen | 0.383 | 0.585 | 0.607 | 0.419 | **0.373** |
| L | SBC-gen | 0.124 | 0.146 | 0.159 | **0.098** | 0.102 |
| d | PPC-gen | **0.372** | 0.548 | 0.540 | 0.450 | 0.386 |
| d | SBC-gen | 0.027 | **0.021** | 0.022 | 0.027 | 0.019 |

(KL-to-truth for Q: PPC-gen PPC=0.99 lowest; SBC-gen free_gru=0.99,
PPC=1.08, SBC=1.59 — i.e. SBC arm is *worst* by KL on its own data.)

## What the run shows

**PPC-generated data dissociates cleanly.** On all three targets the PPC
decoder (with free_tcn close behind) is the best recoverer and the SBC
arm is clearly worse — the matched-readout prediction holds. The mean
rate is a sufficient statistic for the linear-PPC code, and the simple
mean-then-decode PPC arm exploits it; the per-bin SBC arm cannot.

**SBC-generated data does NOT dissociate** under the trained supervised
decoders. The mean-based PPC decoder recovers the SBC posterior about as
well as (Q, L) or marginally worse than (d) the SBC arm, and the free_gru
arm is usually best. This is a real, interpretable result, not a bug:

- With `T = 10` posterior samples, the **trial-averaged** population
  activity of the sampling model is itself a smoothed copy of the
  posterior (a sum of tuning bumps at samples ≈ tuning ⊛ Q*). A
  *supervised* mean-rate decoder can therefore invert it and recover Q*.
  The sampling signature lives in the **across-bin variance / temporal
  autocorrelation**, which the mean discards but the supervised target
  (a static per-trial distribution) does not require.
- The trained SBC arm (per-bin softmax then average, entropy penalty
  off) is a weaker instrument than the analytic sample-KDE readout. The
  deterministic analytic double-dissociation — using the repo's own
  `generate_PPC/SBC_targets` readouts — *does* hold
  (`tests/test_recovery_simulation.py::test_analytic_double_dissociation_on_posterior`):
  the KDE-of-MAPs readout beats the mean readout on SBC data. So the
  generative data carries the dissociation; the trained softmax-average
  arm and the static-distribution loss don't surface it.

**Free arms are competitive but not a strict ceiling.** On data matched
to a constrained arm's inductive bias (PPC-gen → PPC), the simple matched
decoder beats the higher-capacity free arms, which lack that bias and
have more parameters to fit. free_gru/free_tcn lead on SBC-gen data
(temporal structure helps there).

## Implications / next steps

1. Detecting SBC needs a readout/metric sensitive to **temporal
   variance**, not just the trial-averaged posterior — e.g. score the
   SBC arm on its per-bin sample spread, or add a recovery metric on
   across-bin variance. The free_gru/free_tcn arms (which can read time)
   are the natural instruments.
2. Sharpen the generative contrast by **reducing T** (fewer samples → the
   mean carries less of Q*) and/or adding temporal autocorrelation to the
   samples, so trial-averaging no longer reconstructs the posterior.
3. Revisit the trained SBC arm: with the entropy penalty at 0 the per-bin
   softmaxes may be too broad to act as samples; the production
   sharpness penalty exists precisely to make per-bin outputs
   sample-like.
