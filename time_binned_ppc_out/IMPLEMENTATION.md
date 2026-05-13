# Time-binned PPC — implementation reference

Source: `nn_decoder/time_binned_ppc.py`. This document describes the exact
computation done by each PPC variant, the upstream pipeline that produces
their inputs, and the downstream comparisons against the ideal-observer
(IO) targets.

---

## 1. Pipeline overview

For each mouse and each `time_window ∈ {full, half}`:

1. **Load** (`utils_v26.load_vr_export`) — returns
   `activities_raw : (n_neurons, n_trials, n_xG)` over `xG ∈ [0, 2] s` at
   native 50 ms bins, plus IO targets and trial metadata.
2. **Permute** to `(n_trials, t_bins_native, n_neurons)` at the call site
   in `time_binned_ppc.main`.
3. **Window + bin** (`utils_v26.apply_temporal_binning`,
   `time_window=…`, `bin_size_ms=100`, `base_bin_ms=50`):
   - `full`  → keep all 40 native bins (0–2 s) → 20 × 100 ms bins.
   - `half`  → keep last 20 native bins (1–2 s) → 10 × 100 ms bins.
   - Downsampling within each 100 ms group uses `np.nanmean` over the two
     constituent 50 ms bins, so `binned[trial, t, neuron]` is the **mean
     deconvolved-spike rate** in that 100 ms bin (not a spike count).
4. **Fit tuning templates** from the binned activity (§2).
5. **Compute each PPC variant** (§3).
6. **Compare against IO** (§4).

Output: `(n_trials, S)` distributions over `S_GRID = np.arange(0, 91)`
per variant, per mouse, per window.

---

## 2. Tuning templates

Both template kinds are fit using `fit_template_from_rates(rates_2d,
trials, s_grid)`:

For each unique orientation `o` in `trials['orientation']`:

- `ori_mask = (orientation == o)`
- `max_c   = max(contrast[ori_mask])`
- `min_d   = min(dispersion[ori_mask & contrast == max_c])`
- "Easy" trials for `o` = `ori_mask & (contrast == max_c) & (dispersion == min_d)`
- Empirical template entry `templates_emp[o_idx, :] = np.nanmean(rates_2d[easy_mask, :], axis=0)`

Then linearly interpolate the empirical (n_unique_oris, n_neurons) array
along the orientation axis onto `s_grid = [0, 1, …, 90]` and clip values
below `1e-6` (to keep `log f` finite). Final shape `(91, n_neurons)`.

### 2a. Stationary template

One call to `fit_template_from_rates(binned.mean(axis=1), trials)` →
`template_stat : (91, n_neurons)`. Each neuron's response is averaged
over the entire `t_bins` window before template fitting, so the
template is a single tuning curve per neuron.

### 2b. Bin-specific templates

`fit_templates_per_bin(binned, trials)` calls
`fit_template_from_rates(binned[:, t, :], trials)` for every
`t ∈ {0, …, t_bins-1}` → `templates_tv : (t_bins, 91, n_neurons)`. Each
slice along axis 0 is the easy-trial-mean tuning curve estimated from
only the activity in that 100 ms bin.

---

## 3. The three PPC variants

All three return both a **likelihood** `L : (n_trials, 91)` and a
**posterior** `P : (n_trials, 91)`. The prior is the same Go/No-Go
prior the IO fitting uses
(`parameter_recovery_edit.m::get_prior`,
`ideal_observer_hierarchical_fitting_edit.m`, `prior_strength = 3`):
a doubled-angle mixture of two von Mises components centred at 0° and
90°, with κ = 3. `_prior_bimodal(s_grid, kappa=3.0)` implements

  c_μ(s) ∝ exp(κ · cos(2·deg2rad(s) − 2·deg2rad(μ))),  μ ∈ {0°, 90°}
  π(s)    = (c₀ + c₉₀) / Σ_s (c₀ + c₉₀)

renormalised on `s_grid`. The `1 / (2π · I₀(κ))` normaliser of each
von Mises component is omitted because it cancels in the discrete
renormalisation. At κ = 3 the prior has peak/trough ratio ≈ 10.07
(p(0°)/p(45°)).

The softmax over `s` uses `_softmax_over_s`: subtract `nanmax` along the
last axis, `np.exp`, then divide by `nansum`. This makes the resulting
distributions identical to applying any constant `s`-independent offset
to the log-likelihood.

Convention: `r_{t,n}` = 100 ms-binned mean rate for trial `i`, bin `t`,
neuron `n`. `f_n(s)` = template's expected rate for neuron `n` at
orientation `s`. `S = 91`, `N = n_neurons`, `T = n_bins`.

### 3a. TimeAvg — `ppc_time_avg(activities, template_stationary, prior)`

Time-averaged rates with a stationary template. Numerically equivalent
(verified to ≤ 4e-11) to `utils_v26.generate_PPC_targets(activities,
templates, s_grid, beta=1.0, prior_type='bimodal')`.

```
r̄_n          = (1/T) Σ_t r_{t,n}                            # nanmean over t
LL_avg(s)    = Σ_n  r̄_n · log f_n(s)  −  f_n(s)
L_avg        = softmax_s( LL_avg )
P_avg        = softmax_s( LL_avg + log π )
```

Implementation (lines 142–158):

```python
r_mean = np.nanmean(activities, axis=1)                       # (n_trials, N)
log_f  = np.log(f)                                            # (S, N)
LL     = np.nansum(r_mean[:, None, :] * log_f[None, :, :]
                   - f[None, :, :], axis=-1)                  # (n_trials, S)
L      = _softmax_over_s(LL)
P      = _softmax_over_s(LL + np.log(prior + 1e-12)[None, :])
```

### 3b. TimeInt-stationary — `ppc_time_int_stationary(activities, template_stationary, prior)`

Bin-wise Poisson log-likelihoods summed across bins with the **same**
stationary template. The closed form is a temperature-sharpened TimeAvg:

```
R_n         = Σ_t r_{t,n}                                     # nansum over t
LL_int(s)   = Σ_t Σ_n  r_{t,n} · log f_n(s) − f_n(s)
            = R_n · log f_n(s)  −  T · f_n(s)                  # per-s
            = T · ( Σ_n  r̄_n · log f_n(s) − f_n(s) )
            = T · LL_avg(s)
L_int_stat  = softmax_s( LL_int )  =  softmax_s( T · LL_avg )
P_int_stat  = softmax_s( LL_int + log π )                       # NB: log π once, not T·log π
```

Implementation (lines 161–182):

```python
R    = np.nansum(activities, axis=1)                           # (n_trials, N)
LL   = (np.nansum(R[:, None, :] * log_f[None, :, :], axis=-1)
        - t_bins * np.nansum(f, axis=-1)[None, :])             # (n_trials, S)
L    = _softmax_over_s(LL)
P    = _softmax_over_s(LL + np.log(prior + 1e-12)[None, :])
```

Consequences:

- `L_int_stat(s) = L_avg(s)^T / Σ_s' L_avg(s')^T`. Same MAP as TimeAvg,
  but T-fold sharper (`Var(L_int_stat) ≈ Var(L_avg) / T`).
- The posterior identity does **not** hold: TimeInt applies the prior
  once, while `softmax(T · log P_avg)` would apply it T times. The
  stationary identity check in `fig5_stationary_sanity_*.png` is plotted
  on **likelihoods** and lies exactly on the diagonal.

### 3c. TimeInt-timevarying — `ppc_time_int_timevarying(activities, templates_per_bin, prior)`

Bin-wise Poisson log-likelihoods summed with **bin-specific** templates.
This is the only variant that is not reducible to TimeAvg.

```
LL_int_tv(s) = Σ_t Σ_n  r_{t,n} · log f_t,n(s)  −  f_t,n(s)
L_int_tv     = softmax_s( LL_int_tv )
P_int_tv     = softmax_s( LL_int_tv + log π )
```

Implementation (lines 185–202), with `templates_per_bin` of shape
`(T, S, N)`:

```python
log_f  = np.log(templates_per_bin)                              # (T, S, N)
term_r = np.einsum('itn,tsn->is', activities, log_f)            # (n_trials, S)
term_f = np.sum(templates_per_bin, axis=(0, 2))                 # (S,) = Σ_t Σ_n f_t,n(s)
LL     = term_r - term_f[None, :]
L      = _softmax_over_s(LL)
P      = _softmax_over_s(LL + np.log(prior + 1e-12)[None, :])
```

Notes:

- `term_f` does not depend on the trial because the templates are fit
  per bin but pooled across trials (easy-trial-mean within each bin).
- This collapses to TimeInt-stationary iff all bin-specific templates
  are identical to the stationary template, which they are in
  expectation under truly stationary tuning but **not** in finite
  samples (per-bin templates are noisier than the all-bins-pooled
  stationary template).

---

## 4. Comparing against IO

For each variant we record, per trial:

- **Distribution moments** (`dist_mean_var`, linear over `s_grid`):
  `PPC_Lik_Mu`, `PPC_Lik_Var`, `PPC_Post_Mu`, `PPC_Post_Var`.
- **Posterior entropy** (`dist_entropy`): `PPC_Post_H`.
- **KL divergences** (`dist_kl`): `KL(PPC_P || IO_post)`,
  `KL(IO_post || PPC_P)`, and the analogous pair for likelihoods.
- **PCA-weighted distance** (§4a): `PCA_dist_post`, `PCA_dist_lik`.

Aggregated per-mouse Pearson/Spearman correlations vs IO summaries are
written to `correlation_summary.csv`. Per-condition (orientation,
contrast, dispersion) breakdowns are in `condition_breakdown.csv`.

### 4a. PCA-weighted distance

Reproduces exactly the production training loss in
`nn_decoder/neural_network_classifier_v26.py` (function
`custom_loss_all_H`, PCA branch) and the basis used by
`run_experiment_v26.py`.

**Basis construction** (`fit_pca_basis`):

1. Group trials by unique `(orientation, contrast, dispersion)`.
2. Average the IO target distribution within each group →
   `(n_conditions, 91)` matrix.
3. Fit `sklearn.decomposition.PCA()` (no n_components arg → full basis).
4. Return `(pca.components_, pca.explained_variance_ratio_)`.
   If fewer than 3 conditions exist, return `(None, None)` and fall
   back to plain MSE.

A separate basis is fit per mouse: one on `targets_perc`, one on
`targets_lik`.

**Loss** (`pca_weighted_distance`):

```
proj_d   = pred   @ pcs.T                  # (n_trials, n_components)
proj_t   = target @ pcs.T
distance = Σ_k  evar_k · (proj_d_k - proj_t_k)²  ×  100
```

The factor of 100 matches the production loss scale and has no effect
on rank ordering between variants.

---

## 5. Outputs

| Path | Contents |
|------|----------|
| `time_binned_ppc_out/per_trial.csv` | One row per (variant, trial), all metrics above |
| `time_binned_ppc_out/correlation_summary.csv` | Per-(mouse, variant, window) Pearson/Spearman r |
| `time_binned_ppc_out/pca_distance_summary.csv` | Per-(window, variant) mean ± SEM PCA distance across mice |
| `time_binned_ppc_out/condition_breakdown.csv` | Per-(metric, axis, level, window, variant) mean ± SEM across mice |
| `time_binned_ppc_out/figures/fig1_examples_{full,half}.png` | Example trial distributions per mouse (low/mid/high IO var) |
| `time_binned_ppc_out/figures/fig2_correlation_bars.png` | r(PPC summary ~ IO summary) bar grid |
| `time_binned_ppc_out/figures/fig3_hexbin_{full,half}.png` | Pooled hexbin scatter of PPC vs IO summaries |
| `time_binned_ppc_out/figures/fig4_kl_histograms.png` | KL(PPC \|\| IO_post) histograms across variants × windows |
| `time_binned_ppc_out/figures/fig5_stationary_sanity_{full,half}.png` | Numerical check: H(L_int_stat) ≡ H(softmax(T·log L_avg)) |
| `time_binned_ppc_out/figures/fig6_pca_distance_bars.png` | Production PCA-weighted distance bar chart |
| `time_binned_ppc_out/figures/fig7_pca_distance_vs_uncertainty.png` | r(PCA_dist_post ~ IO_Post_Var) per-mouse mean |
| `time_binned_ppc_out/figures/fig8_pca_dist_post_by_condition.png` | PCA dist to IO posterior vs ori/contrast/dispersion |
| `time_binned_ppc_out/figures/fig9_pca_dist_lik_by_condition.png` | PCA dist to IO likelihood vs ori/contrast/dispersion |
| `time_binned_ppc_out/figures/fig10_kl_post_by_condition.png` | KL(PPC \|\| IO_post) vs ori/contrast/dispersion |
| `time_binned_ppc_out/figures/fig11_mean_bias_by_condition.png` | PPC posterior mean − IO posterior mean vs condition |

Synthetic tests covering the temperature identity, decoding sanity, and
the per-bin-templates-help case live in
`tests/test_time_binned_ppc.py` (4 tests; run with
`python -m pytest tests/test_time_binned_ppc.py`).
