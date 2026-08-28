# VR_Decoder_Data_Export.mat — data hand-off

Mouse **V1 two-photon calcium imaging** during a **virtual-reality Go/NoGo
orientation-discrimination task**, with a matched **ideal-observer (IO) model**
fit to behaviour. One self-contained MATLAB file holds the neural responses, the
per-trial behaviour/stimulus table, and the IO posteriors used as decoding
targets.

This README describes the file's structure and conventions. Four validation
figures in [`figures/`](figures/) let you confirm the data is sensible before
relying on it (see [Validation figures](#validation-figures)).

---

## At a glance

| | |
|---|---|
| **Subjects** | 6 mice (`Cb15, Cb17, Cb21, Cb22, Cb24, Cb25`) |
| **Neurons** | 651 kept cells total (65–153 per mouse) |
| **Trials** | 4368 total (651–940 per mouse), 4–5 sessions each |
| **Recording** | 2-photon calcium imaging of V1; ΔF/F + deconvolved spike probability |
| **Task** | VR Go/NoGo: lick (go) vs withhold (nogo) by grating orientation |
| **Two epochs** | **grating** (2 s stimulus, binned in **time**) → **corridor** (navigation, binned in **position**) |
| **File** | `VR_Decoder_Data_Export.mat`, MATLAB **v7** (not v7.3/HDF5), ≈ 560 MB |

Per-mouse breakdown:

| Mouse | Neurons | Trials | Sessions |
|-------|--------:|-------:|---------:|
| Cb15  | 65  | 940 | 5 |
| Cb17  | 105 | 651 | 5 |
| Cb21  | 74  | 687 | 5 |
| Cb22  | 153 | 671 | 5 |
| Cb24  | 142 | 721 | 4 |
| Cb25  | 112 | 698 | 5 |
| **Total** | **651** | **4368** | |

---

## Loading

**Python** (file is v7, so plain `scipy` works — no `h5py` needed):

```python
import scipy.io as sio
mat = sio.loadmat("VR_Decoder_Data_Export.mat", simplify_cells=True)
NS  = mat["NeuralStore"]      # list of 6 dicts, one per mouse
TBL = mat["TrialTbl_Struct"]  # dict of length-4368 arrays (pooled trial table)
IO  = mat["IO"]               # ideal-observer fit block
```

`simplify_cells=True` is important — it turns MATLAB structs into nested
dicts/lists and squeezes singleton dimensions.

**MATLAB:**

```matlab
S = load('VR_Decoder_Data_Export.mat');   % S.NeuralStore, S.TrialTbl_Struct, S.IO
ns15 = S.NeuralStore(1);                   % struct for mouse Cb15
```

---

## Top-level structure

The file has three variables:

| Key | What it is |
|---|---|
| `NeuralStore` | Per-mouse **neural responses** + per-mouse trial metadata. Cell/list of 6 structs. |
| `TrialTbl_Struct` | **Pooled trial table** (struct-of-arrays), one row per trial across all mice/sessions, including IO targets. |
| `IO` | The fitted **ideal-observer model** (group + per-animal parameters) that produced the posterior targets. |

---

## `NeuralStore` — per-mouse neural data

`NeuralStore` is a length-6 list; `NeuralStore[i]` is one mouse. Two stimulus
epochs are stored separately:

- **Grating epoch (`G` / "gr")** — the fixed **2-second** window while the
  grating is on screen, binned in **time** at 50 ms → axis `xG` (40 bins,
  0.025–1.975 s). This is the **temporal** axis.
- **Corridor epoch (`C` / "co")** — the variable-length **navigation** phase
  after the grating, binned by **VR position** at 5 cm → axis `xC` (70 bins,
  2.5–347.5 cm). This is the **spatial** axis.

| Field | Shape | Meaning |
|---|---|---|
| `tag` | str | Animal id, e.g. `'Cb15'` |
| `xG` | (40,) | Grating **time** bin centres (s), 0.025…1.975, 50 ms steps |
| `xC` | (70,) | Corridor **position** bin centres (cm), 2.5…347.5, 5 cm steps |
| `Gspk` | (n_trials, n_neurons, 40) | Grating-epoch **deconvolved spike probability**, mean per time bin |
| `Cspk` | (n_trials, n_neurons, 70) | Corridor-epoch deconvolved spike probability, mean per position bin |
| `GdF` | (n_trials, n_neurons, 40) | Grating-epoch **ΔF/F**, mean per time bin |
| `CdF` | (n_trials, n_neurons, 70) | Corridor-epoch ΔF/F, mean per position bin |
| `keep` | (n_neurons,) | **Original ROI indices** of the kept cells (not a 0/1 mask) |
| `nKept` | int | Number of kept neurons (= `n_neurons`) |
| `contrast`, `dispersion`, `stimulus`, `outcome` | (n_trials,) | Per-trial metadata, **row-aligned to the neural trial axis** (see below) |

**Values are raw** (not z-scored). `Gspk`/`Cspk` are mean **inferred spike
probability** per bin (dimensionless, ~0–0.1; *not* a firing rate in Hz);
`GdF`/`CdF` are mean ΔF/F. Spike probability comes from deconvolution of the
FISSA-corrected ΔF traces.

`Cspk`/`CdF` contain **NaNs** at far position bins not reached on a given trial
(corridor length varies) — use `nanmean` and check bin occupancy.

---

## `TrialTbl_Struct` — pooled trial table (4368 rows)

Struct-of-arrays: every field is a length-4368 array (or 4368×K matrix), one row
per trial pooled over all mice and sessions.

**Identity**

| Column | Meaning |
|---|---|
| `animal` | Animal tag (`'Cb15'`…) |
| `session` | Per-animal session index (1–5) |
| `trial` | Within-session trial number |

**Stimulus**

| Column | Meaning |
|---|---|
| `stimulus` | Presented grating orientation, 0–90° (9 levels: 0,15,30,40,45,50,60,75,90) |
| `contrast` | Stimulus contrast: 0.01, 0.25, 0.5, 1.0 |
| `dispersion` | Orientation-ensemble spread (deg): 5, 30, 45, 90. Higher = noisier stimulus → lower IO sensory precision (`kappa = kappa0·contrast^c_power·exp(−d_power·dispersion)`) |
| `theta_deg`, `go_is_vertical` | Raw grating angle; whether the go target is vertical this session |
| `theta_from_go` | **Signed** angle from the **go pole** (rewarded orientation); 0 = go pole |
| `abs_from_go` | `|theta_from_go|`, 0–90° |

**Behaviour**

| Column | Meaning |
|---|---|
| `goChoice` | 1 = licked / go, 0 = withheld / nogo |
| `outcome` | **1 = Hit, 2 = Miss, 3 = False Alarm, 4 = Correct Rejection** |
| `performance` | 1 = correct, 0 = error |
| `preRZ_velocity` | Mean pre-reward-zone running speed |
| `preRZ_licks`, `preRZ_lick_rate` | Pre-reward-zone lick count / rate |
| `confidence` | Behavioural proxy = `zscore(lick_rate) − zscore(velocity)` (continuous, ~ −4…8) |

**Per-trial neural summaries** (`_gr` = grating epoch, `_co` = corridor epoch)

| Column | Meaning |
|---|---|
| `meanAct_gr`, `meanAct_co` | Mean spike probability over the epoch |
| `meandF_gr`, `meandF_co` | Mean ΔF/F over the epoch |
| `logGV_gr`, `logGV_co` | Log generalised variance of the population (covariance-structure summary) |
| `normGV_gr`, `normGV_co` | Generalised variance normalised by activity magnitude |

**Ideal-observer targets** — posteriors over orientation `s` on a **91-point
grid, 0–90°** (the decoding targets):

| Column | Shape | Meaning |
|---|---|---|
| `post_s_marginal` | (4368, 91) | Perceptual posterior P(s\|data), measurement **marginalised out** — the standard target |
| `post_s_given_map` | (4368, 91) | Posterior conditioned on the MAP measurement |
| `L_s_marginal`, `L_s_given_map` | (4368, 91) | Sensory **likelihood** (no prior on s), marginal / MAP variants |
| `decision_posterior` | (4368, 2) | 2-class **[go, nogo]** posterior (= `decision_posterior_marginal`) |
| `decision_posterior_map`, `decision_posterior_marginal` | (4368, 2) | MAP / marginal variants |
| `unc_perceptual`, `unc_decision` | (4368,) | Scalar perceptual / decision uncertainty |

---

## `IO` — ideal-observer model fit

The hierarchical IO model that generated the posterior targets, for provenance /
re-derivation.

- `IO.meta` — `timestamp`, `model_spec` (fitted parameter names, grids, prior),
  `fit_mode` (`'conf_only'`), `fixed_utility`. The orientation grid
  `s_range_deg` is 0:90 (91 bins); the measurement grid `m_range_deg` is 0:180
  (181 bins); `prior_type` is `'Bimodal'` (the go/nogo prior).
- `IO.group` — group-level `params` (6-vector) and `avg_test_nll`.
- `IO.animals` — 6 per-animal entries (`fit`, `pred`, `inferred`,
  `trial_table`, `go_side`). **Note:** these are tagged positionally
  (`Animal_1`…`Animal_6`) in NeuralStore/table order (Cb15, Cb17, Cb21, Cb22,
  Cb24, Cb25), **not** by `Cb*` tag.

---

## Key conventions & gotchas

1. **Joining neural data to the trial table.** For mouse `i`, the neural trial
   axis (`Gspk`/`Cspk` axis 0, and `NeuralStore[i]`'s own
   `contrast`/`dispersion`/`stimulus`/`outcome`) is **row-aligned, in order**, to
   the subset of `TrialTbl_Struct` rows where `animal == NeuralStore[i].tag`.
   Verified element-wise on `stimulus`/`dispersion`/`outcome` for all 6 mice. So:

   ```python
   import numpy as np
   tag  = NS[0]["tag"]                                   # 'Cb15'
   rows = np.asarray(TBL["animal"]).astype(str) == tag   # mask into the pooled table
   Gspk = NS[0]["Gspk"]                                  # (940, 65, 40)
   post = np.asarray(TBL["post_s_marginal"])[rows]       # (940, 91)  IO targets
   # Gspk[k] and post[k] are the same trial k.
   ```

2. **`G` is time, `C` is position.** The grating axis `xG` is **seconds**
   (2 s window); the corridor axis `xC` is **centimetres**. They are *not* the
   same kind of axis — don't concatenate them naively.

3. **Contrast is encoded as `0.99` in `NeuralStore` but `1.0` in
   `TrialTbl_Struct`** for full-contrast trials (the only difference; rows are
   otherwise identical). Snap `contrast[contrast > 0.9] = 1.0` to unify, giving
   four clean levels {0.01, 0.25, 0.5, 1.0} for all mice.

4. **Population activity does *not* increase monotonically with contrast** in the
   full-window grand mean — it actually *decreases* slightly at high contrast.
   This is **expected here, not a data fault**: high-contrast trials are easy and
   ridden through with lower arousal/running, the 2 s mean is dominated by a late
   anticipatory rise, and the full-contrast ensemble is pole-heavy. The genuine
   stimulus drive shows up **only when measured properly** — in the **onset
   window (0–300 ms), at each cell's preferred orientation** — where the
   contrast-response is cleanly **saturating** (see `fig2`). Condition carefully
   before reading contrast effects off the population mean.

5. **Orientation is ~180°-periodic and folded to 0–90°.** P(go) peaks at the go
   pole (`theta_from_go = 0`) and falls toward the orthogonal nogo pole at ±90°,
   so the psychometric is **bell-shaped in signed orientation**, not a monotonic
   sigmoid (see `fig4`).

6. **Units:** `Gspk`/`Cspk` are mean inferred spike *probability* per bin
   (≈0–0.1, dimensionless), not Hz. ΔF/F can be negative.

---

## Validation figures

In [`figures/`](figures/) (PNG for preview + SVG for vector). Regenerate with
`python make_validation_figures.py` (reads `../../data/VR_Decoder_Data_Export.mat`).

| Figure | Confirms |
|---|---|
| `fig1_orientation_tuning` | V1 cells are **orientation-tuned**: example tuning curves + a population heatmap (651 cells sorted by preferred orientation → clean diagonal). |
| `fig2_grating_response` | A time-locked **evoked transient** in the grating window, and a **saturating contrast-response** (onset window, preferred orientation, 474 driven cells). |
| `fig3_corridor_position` | **Corridor (spatial) responses tile track position** — neurons sorted by peak position form a sequential diagonal (example mouse Cb15). |
| `fig4_behaviour_and_io` | A sensible behavioural **psychometric** (P(go) peaks at the go pole) and **IO posteriors** that peak at the true stimulus on the 0–90° grid. |

---

## Provenance

Built from per-session VR + imaging data by
`preprocessing/VR_multi_animal_analysis.m` → `createUnifiedSessionData.m`; IO
targets from `ideal_observer/ideal_observer_hierarchical_fitting_v2.m`
(hierarchical fit, refreshed into the export by `refresh_IO_in_export.m`).
