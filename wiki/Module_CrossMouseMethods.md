# Cross-mouse methods in `nn_decoder/dim_reduction_explore.py`

This document explains the four cross-mouse comparison methods the exploratory dim-reduction script supports, plus the per-mouse archetype analysis that doubles as a cross-mouse comparison because every mouse defines the same coordinate system intrinsically.

The fundamental challenge is that neural recordings differ across mice — different units, different drift, different per-neuron tuning. A PCA basis fit on mouse A is not the same basis as PCA on mouse B, so trajectories in PC space are not directly comparable across animals. Each method below answers "how do I project all mice into the same coordinate system?" in a different way, with different assumptions and different things they preserve.

## 1. Procrustes alignment (Style E `procrustes`)

**What it does.** Each mouse's condition-averaged trajectories are first projected into its own 2D or 3D PCA basis. One mouse (the one with the most trials that has all the required condition levels present) is picked as reference. Every other mouse's trajectory set is then rigidly rotated, reflected, and uniformly scaled to minimise the Frobenius distance to the reference on the matched condition×xG points. The reference itself is also centered and scaled by the Procrustes idiom so all mice end up in the same "standardised" frame.

**Formally.** Given a reference matrix `R_ref` and a mouse matrix `R_m`, both `(n_xG × n_levels, k)` after stacking, scipy's `procrustes(R_ref, R_m)` returns a transformation `T(R_m) = c · R_m · Q` where `Q` is orthogonal (rotation + reflection) and `c` is a positive scalar (uniform scale) chosen to minimise `||R_ref_std − T(R_m)||_F`. The reference is also standardised (mean-centered and scaled to unit Frobenius norm) for the comparison. We then apply `T` to that mouse's trajectories and overlay all mice in the reference frame.

**Preserves.** Distances and angles *within* each mouse's trajectory geometry (it's a rigid transformation up to global scaling). Two trajectories that diverged in mouse A still diverge by the same relative amount after alignment.

**Does NOT preserve.** Absolute units — Procrustes throws away absolute scale. The aligned space has no intrinsic units; only the geometry is meaningful. Also, the alignment is data-dependent: change which mouse is the reference, and the canonical frame rotates.

**When to use.** When you want to *visualise* whether the same structural relationships (e.g. Go vs NoGo condition trajectories) appear across mice. It is fundamentally a visualisation tool, not a quantitative cross-mouse decoder.

**Failure modes.** Few common condition levels across mice (degenerate alignment). Different mice having structurally different geometries (Procrustes will force-fit and the result will look noisy).

## 2. Task-axis regression (Style E `taskaxis`)

**What it does.** For each mouse, regress trial-mean neural activity against a panel of task variables (true_orientation, P(Go), H_dec, sd_perc, signed_contrast, signed_concentration). For each task variable `y`, ridge regression fits `y ≈ X · w + b`, where `X` is `(n_trials, n_neurons)` and `w` is the **encoding direction** for that variable in neural space. Stack the unit-normalised encoding directions across task variables → `W ∈ ℝ^(n_neurons × k)`. Project: `Z = X · W` gives every trial a `(k,)`-coordinate in a space whose axes are *defined* by task variables.

**Formally.** For each task variable `y_j`:
```
w_j = argmin_w ‖y_j − X·w‖²₂ + α‖w‖²₂        (Ridge with α = 1.0)
ŵ_j = w_j / ‖w_j‖₂
W = [ŵ_1, ŵ_2, …, ŵ_k]
Z_i = X_i · W            (trial i's coordinate in shared task space)
```

**Preserves.** Cross-mouse comparability *by construction* — axis 0 is "the orientation-encoding direction", axis 1 is "the P(Go)-encoding direction", etc. These mean the same thing in every mouse, so distances along each axis are interpretable as "how much this trial agrees with that task variable, in this mouse's neural population". Different mice land in literally the same coordinate system; you can overlay them without any further alignment.

**Does NOT preserve.** Geometric structure orthogonal to task axes. Anything in the neural population that *doesn't* correlate with the listed task variables disappears in this projection. The axes are not orthogonal to each other in general (task variables are correlated), so distances are not Euclidean — they're "task-variable-weighted".

**When to use.** When you want a shared coordinate system that has *meaning*, not just visual alignment. Best for asking "does the relationship between orientation and decision look the same across mice?" rather than "does the overall geometry look the same?".

**Failure modes.** Multi-collinear task variables — if signed_contrast and p_go are highly correlated, their regression weights are unstable and the axes become poorly separated. The script uses `α = 1.0` ridge regularisation to dampen this. If a task variable is constant within a mouse (e.g., the mouse only ever saw one contrast level), its axis is degenerate; that mouse is dropped from that axis.

## 3. CEBRA multi-session continuous (Style D)

**What it does.** CEBRA is a contrastive learning method that learns a low-dimensional embedding such that points with similar conditional values are pulled together and dissimilar points pushed apart. In **multi-session mode**, CEBRA trains one shared "trunk" producing a common output embedding, but uses subject-specific input "head" encoders so each mouse's variable-dimensional `(n_neurons,)` input maps to the same `(output_dim,)` embedding.

**Formally.** For each mouse `s`, a session-specific encoder `f_s: ℝ^(n_neurons_s) → ℝ^d` shares structure with all other encoders in the loss. The contrastive objective uses the continuous label `y` (in our script: `signed_contrast`) as the conditional — pairs of samples with similar `y` are positive pairs, dissimilar are negatives. Training minimises an InfoNCE-style loss:
```
L = -log [ exp(sim(z_pos, z_anchor)/τ) / Σ_k exp(sim(z_k, z_anchor)/τ) ]
```
where the similarity is cosine. The result is a single embedding space in which every mouse's points live.

**Why we switched from discrete to continuous.** Discrete-label multi-session CEBRA requires aligned label distributions across sessions (every session must see most labels), and the discrete objective is brittle when label counts per session are small. Continuous labels (signed_contrast spans Go→ambiguous→NoGo on a single real axis) work better because every trial contributes a graded label, and the contrastive structure has more pairs to compare.

**Preserves.** The relationship between neural activity and the conditional variable (signed_contrast in our case). Trials with similar signed_contrast end up close in the embedding regardless of which mouse they came from. The embedding has no intrinsic geometric meaning beyond "samples close in this space have similar conditional values".

**Does NOT preserve.** Anything orthogonal to the conditional — the embedding only learns structure that the conditional discriminates. If two trials differ in choice but agree on signed_contrast, they will not be separated by CEBRA. (To recover choice structure, you'd colour the embedding by choice and look for residual within-cluster separation, or retrain with choice as the conditional.)

**When to use.** When you want a *learned* alignment that respects a specific behavioural variable, robust to different per-neuron tuning across mice. CEBRA is also strong for embedding visualization in 3D for label-driven cluster structure.

**Failure modes.** Multi-session CEBRA can fail outright (NaN loss, optimisation divergence) when one mouse's data is much larger or has very different statistics than the others. The script wraps the fit in a try-except and falls back to per-mouse single-session CEBRA, which still gives interpretable per-mouse embeddings but loses the shared-coordinate property. The figure title declares which mode succeeded.

## 4. Archetype similarity (Style F)

**What it does.** For each mouse, define **archetype trajectories** as the mean z-scored neural trajectory of "easy" trials — top-quartile contrast AND bottom-quartile dispersion — split by stimulus side (Go-side `stimulus < 45°` vs NoGo-side `stimulus ≥ 45°`). Each archetype is a `(n_neurons, n_xG)` trajectory representing "what activity looks like when the mouse should know what to do." For every trial, compute its **cosine similarity** to each archetype at each xG bin. Each trial then has a `(n_xG, 2)` trajectory in the **shared 2D plane** `(sim_NoGo, sim_Go)`.

**Formally.** Per mouse:
```
A_Go   = mean_{i ∈ easy_Go} Z_i      ∈ ℝ^(n_neurons × n_xG)
A_NoGo = mean_{i ∈ easy_NoGo} Z_i    ∈ ℝ^(n_neurons × n_xG)

For each trial i and each xG bin t:
   sim_i,t,k = ⟨Z_i,:,t , A_k,:,t⟩ / (‖Z_i,:,t‖ · ‖A_k,:,t‖)
```
The result is a `(n_trials, n_xG, 2)` similarity tensor.

**Why this gives cross-mouse comparability.** Every mouse defines its own archetypes from its own easy trials, so the **basis is defined intrinsically** by behavioural semantics: `(1, 0)` always means "looks like an easy NoGo in *this* mouse's brain"; `(0, 1)` always means "looks like an easy Go". These semantics are identical across mice even though the underlying neural populations are different. The diagonal `sim_Go = sim_NoGo` represents ambiguity (the activity is equally close to both archetypes).

**Preserves.** The behavioural semantic of "how close to the easy-trial reference patterns is this trial?" Two mice that show the same temporal evolution from ambiguous to decision-committed will produce similar curves in this plane.

**Does NOT preserve.** Anything the archetypes don't capture. If both archetypes happen to share a common drift mode (e.g. uniform population activity rising over xG), that mode contributes to both similarities equally and is invisible in the difference `sim_Go − sim_NoGo`. Also, the absolute similarity values depend on noise in the archetypes (low n_easy → noisy archetype → smaller absolute similarity).

**When to use.** When you have a clean "easy reference" and want to ask **how do trials of various conditions evolve relative to that reference, decision-axis-wise?** Particularly powerful for asking "when in the trial does choice information emerge?" — the `sim_Go − sim_NoGo` curve gives a per-mouse decision-axis time course, and grouping by choice or certainty reveals when those groups diverge.

**Failure modes.** Too few easy trials → noisy archetype. The script requires at least `ARCHETYPE_MIN_TRIALS = 5` easy trials per side and falls back to skipping a mouse otherwise. If contrast and dispersion don't co-vary in the experimental design as expected, the "easy" filter may select an empty set or a biased subset.

## Method comparison

| Method            | Coordinate system            | Cross-mouse comparable? | Preserves geometry? | Preserves task semantics? | Needs alignment data? |
|-------------------|------------------------------|-------------------------|---------------------|----------------------------|------------------------|
| Procrustes        | reference mouse's PC basis   | After alignment         | Yes (rigid)         | Indirectly                 | Matched condition labels |
| Task-axis         | shared task-variable axes    | By construction         | No (oblique)        | Yes                        | Same task variables    |
| CEBRA continuous  | learned latent (3D)          | If multi-session OK     | Only along label    | Along conditional only     | Same conditional       |
| Archetype sim.    | (sim_NoGo, sim_Go)           | By construction         | No (cosine)         | Yes                        | Easy trials available  |

## Practical recipe

If you want **a quick visual sanity check** that mice show the same structural patterns: Procrustes. Cheap, deterministic, but only visual.

If you want **shared coordinates with task meaning**: task-axis regression. The axes are *defined* to mean specific task variables, so distances along each axis are interpretable across mice.

If you want **the cleanest decision-axis time course**: archetype similarity. The `(sim_NoGo, sim_Go)` plane gives every mouse the same 2D coordinate system intrinsically, and the diagonal-vs-off-diagonal structure directly visualises decision commitment.

If you want **a learned nonlinear embedding** that automatically aligns mice by a chosen behavioural variable: CEBRA multi-session continuous. Most flexible but slowest and most opaque; results depend on hyperparameters (`max_iterations`, `output_dimension`, `learning_rate`).

In practice, run all four for any question worth answering. If two methods disagree about whether mice share structure, the disagreement itself is informative (it tells you which assumption is failing).
