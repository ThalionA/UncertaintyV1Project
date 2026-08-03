# DATA_MAP — how the decoder data is organised

Canonical reference so analyses don't re-probe the `.mat`/`.pt` structure every time.
**Update this when the layout changes.** Verified 2026-06-14 against `loss_comparison_v1`
and `lambdaH_sweep_*`.

## On-disk layout

```
nn_decoder/results/<run_name>/<cell>/<split>.mat          # decoded + target posteriors, per mouse
nn_decoder/results/<run_name>/<cell>/checkpoints/<mouse>_<split>.pt   # weights, X_test, history
```
Everything under `results/` and `figures/` is **gitignored** (only code is committed).

### `<run_name>`
- `loss_comparison_v1` — the canonical **matched** grid (5 losses, identical hyperparams except the loss). Q&L × {half,full} × {50,100}ms × 6 mice × 3 splits. **Both archs** per run.
- `hidden_ablation_h{4,8,16,32,64}` — width ablation (B1).
- `lambdaH_sweep_entlam<λ>` — the λ_H sweep (2026-06-14); one full grid per λ_H value.
- `wm3`, `wm3_flatevar`, `wm3_shape{1,10,30}`, `wm3_alpha{0p15,0p3,0p5}` — PCA-loss peakiness controls (evar/shape/alpha).
- Older/exploratory: `new_loss_sweep_may27`, `loss_sweep_h10_val_2026_05_27`, `neuron_scaling`, `post_fix_loadings_2026_05_17`, `production_full_targets_alltrials_v1`, `kl_js_entropy_sweep_v1`.

### `<cell>` naming  →  `<target>_<loss>_<window>_<binms>`
e.g. `Q_KL_half_100ms`. **Quirk:** the pooled PCA cell in `loss_comparison_v1` is `Q_PCA_half_100ms_all` (trailing `_all`). Glob `Q_<loss>_half_100ms*` to be safe.
Run-name suffixes (isolated runs): `_h<H>`, `_flatevar`, `_shape<λ>`, `_alpha<α>`, `_entlam<λ>` (dots → `p`, e.g. `_entlam0p003` = 0.003).
- `<target>` ∈ {Q (perceptual posterior), L (likelihood), …};  `<window>` ∈ {half, full};  `<binms>` ∈ {50, 100}.
- `<split>` ∈ {`stratified_balanced` (default), `generalize_contrast`, `generalize_dispersion`}.

## `.mat` structure  (`scipy.io.loadmat(f, simplify_cells=True)['results']`)

`results` is a dict keyed `mouse_0 … mouse_5` (**n = 6**). Per mouse:
- `fit_loss`, `entropy_penalty` — dicts `{'spat':…, 'temp':…}`; `entropy_penalty` saved = `λ_H · mean_t H(per-bin)` (temporal only; spatial 0).
- `trials` — per-trial **stimulus** descriptors (loss-invariant): `orientation` (**linear 0–90°**, ~9 discrete levels; 0/90 = task references, IO often **bimodal/broad near 45°**), `dispersion` (~4 levels, ↑ = more uncertain), `contrast` (~4 levels, ↑ = less uncertain).
- `trial_cats`, `Weights`.
- `Dist` — the posteriors. Keys: `spat`, `spat_shf`, `temp`, `temp_shf`, `pcs` (91×91), `explained_var` (91,).
  - `Dist[arch]` (arch ∈ {`spat`=PPC, `temp`=SBC/sampling}):
    - `decoded` — **(n, 91)** time-averaged posterior (the deployed prediction).
    - `decoded_samp` — **(n, 91, T)** the **per-bin (instantaneous) distributions** (T=10 bins at 100ms/half). `decoded_samp.mean(axis=2) == decoded` (verified). **axis 1 = 91 orientation categories, axis 2 = time bins.**
    - `target` — (n, 91) the IO target posterior (loss-invariant; **mean entropy ≈ 3.71 nats**, max ln 91 = 4.51).
    - `full_decoded` — (2n, 91) (test + held-out / both halves; rarely needed).
  - `Dist[arch+'_shf']` — the **shuffle-fit** control (net trained on label-shuffled data); `_shf['target']` ≈ the real target (allclose). Same sub-keys.
- Posterior grid: 91 bins = orientation 0–90° in 1° steps (`s_grid = arange(0,91)`).

## `.pt` checkpoint  (`torch.load(pt, weights_only=False)`)

Top-level dict `{'spat':…, 'temp':…}`. Each arch:
- `state_dict` — `layers.0.weight/bias` (W_in, H×N_neurons / H), `layers.1.weight/bias` (W_out, 91×H / 91). Single hidden layer in the standard config.
- `X_test` — **(n, T, n_neurons)** held-out neural inputs (e.g. 470×10×65). Spatial integrates over T (`mean`), temporal feeds per-bin.
- `pred_probs`, `model_params` ({input_size, hidden_sizes, output_size, activation_function}), `model_type` ({'ppc','sampling'}), `pcs`, `explained_var`, `loss_func`, `entropy_lambda`.
- `history` — `{train_total_loss, train_fit_loss, train_entropy_pen, train_pca_yardstick, val_total_loss, weight_norms (epochs×4), snapshot_epochs, state_dicts (per-snapshot weights, epoch 0 = init)}`.

## `data/VR_Decoder_Data_Export.mat` — the **`IO`** struct  (verified 2026-07-29)

Top level is `{'IO', 'NeuralStore', 'TrialTbl_Struct'}`. `TrialTbl_Struct` is what `utils.load_vr_export` reads (targets = `post_s_marginal`); the **`IO`** struct is the fitted ideal observer itself and is what you need to open the target up rather than take it as given.

- `IO.meta.model_spec.fit_params` — `['kappa_amp','c_power','d_power','vel_slope','vel_intercept','vel_std']`; `fit_mode = 'conf_only'`.
- `IO.meta.model_spec.fixed_params` — `s_range_deg` (91), `m_range_deg` (**181**), `prior_type='Bimodal'`, `prior_strength=3`, `kappa_min=1`, `decision_beta=1`.
- `IO.animals` — **(6,), ordered to match `mouse_0..5`** even though the tags don't (`Animal_1..6` vs NeuralStore's `Cb15/17/21/22/24/25`). Each animal holds 5 **more** trials than `TrialTbl_Struct` does for the same mouse (TrialTbl drops the last trial of each session). Confirmed by matching every decoder target to its nearest IO row (max L1 5e-8), not by the tags.
  - `.data` — `orientation` (= `abs_from_go`), `contrast`, `dispersion`, `choices`, `conf_licks`, `conf_vel`, `n_trials`.
  - `.fit.full_params` — the per-animal fitted params **plus** the fixed ones, as a flat dict. Use this, not the group vector.
  - `.inferred` — `post_s_marginal` (n,91; **this is the training target**), `post_s_given_map`, `L_s_marginal`, `L_s_given_map`, `perceptual`, `decision`, and **`m_posteriors` (n,181)**.

**`m_posteriors` is the key field: it makes the target's mixture structure exact.** The target is a marginal over the latent measurement, and these are its weights, so
`post_s_marginal[t] == sum_m m_posteriors[t,m] * p(s|m, kappa_t)` **to ~1e-11** — rebuild `p(s|m,kappa)` with `ideal_observer/io_hmm/io_core` (`kappa_for_trial`, `prior_bimodal(grids, 3.0)`, `posterior_s_given_m`) from `.fit.full_params`. Only ~8 unique kappa per animal (kappa depends on contrast and dispersion only), so vectorise per kappa group. Worked example + the derived noise variance: `diagnostics/io_noise_variance.py`.

Caveat: `m_posteriors` is a **posterior** over m, not the generative `p(m|s,kappa)` — it differs from the von Mises by 2.3e-2 and varies between trials in the same (ori, contrast, dispersion) cell. Consistent with the `conf_only` velocity readout; the exact link was not reproducible from the exported params. Irrelevant to the decomposition, which is exact against whatever the weights are.

## Compute conventions
- **Scoring a loss** (matches training exactly): `cross_loss_eval._eval_one(decoded, target, metric, pcs, evar)` → mean per-trial loss; `metric ∈ {PCA, CE, KL, JS, Wasserstein}`. PCA needs `pcs`/`evar`.
- **Skill** (scale-free): `loss / shuffle_loss` (Dist[arch+'_shf']); `<1` beats chance, `1` = chance. Per mouse then `_agg` (mean ± sem over 6).
- **Stricter null**: predict-mean (`np.tile(target.mean(0), …)`) — the optimal constant; see memory [[shuffle-control-nulls]].
- **Forward** (`nn_classifier.get_model_probabilities` / line ~298): spatial = `softmax(MLP(mean_t X))`; temporal = `mean_t softmax(MLP(X_t))`. Killing `W_in` → constant `softmax(MLP(0))` for both archs.
- **Entropy** (nats): `H(p) = -Σ p log p` over axis = 91 categories.

## Gotchas (see GOTCHAS.md for the full list)
- `decoded` is the **time-average**; per-bin lives in `decoded_samp` (don't `.max(1)` on a 3-D array by mistake).
- Orientation is **linear** 0–90°, not circular.
- pytest segfaults under multithreaded BLAS on macOS → `OMP_NUM_THREADS=1`.
- `KLs` field removed post-2026-05-20 (was entropy-inflated for temporal); use `fit_loss`/recompute.
