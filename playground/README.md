# Similarity Framework — Figure Code

Code that generates the six figures embedded in `Similarity_Framework.md`.

## Layout

```
similarity_framework/
├── framework_common.py        shared palette, simulation params, helpers
├── fig01.py                   Bundle archetypes in V1 state space
├── fig02.py                   Cosine as projection onto S^{N-1}
├── fig03.py                   Pointwise SI(t) traces by difficulty
├── fig04.py                   DDM decomposition (v / s / s_v / RT)
├── fig05.py                   Asymmetric strategy (single-archetype limit)
├── fig06.py                   Bayesian framing (vMF / prior / posterior)
├── run_all.py                 orchestrator
└── README.md
```

## Usage

Requires: `numpy`, `matplotlib`, `scipy` (for `i0` in fig06 only).

```bash
# Run everything to ./figures/
python run_all.py

# Or write to the vault directly
SIMFRAME_OUTDIR=~/vault/attachments/similarity_framework python run_all.py

# Or run a single figure
python fig03.py
```

Each `figXX.py` is self-contained — it imports the data from `framework_common.py`
and writes one PNG. Edit the rcParams / palette / DPI in `framework_common.py` to
restyle all figures at once.

## Reproducibility

All randomness uses `np.random.default_rng(seed)` with fixed seeds defined in
`framework_common.py` (trajectory bundles: seeds 100+k for L, 200+k for R) and
in the individual figure scripts (RT samplers etc.). Running the scripts twice
produces byte-identical output up to matplotlib's font-rendering noise.

## Common parameters

Defined in `framework_common.py`:

- `T_TRIAL = 1.0 s`, `DT = 0.005 s` → 200 timesteps
- Two endpoints: `END_L = (-2.2, 1.6)`, `END_R = (2.4, -0.3)`
- Baseline: `START = (0.6, 0.5)`
- Bundle size: `K = 14` exemplars per class
- Per-exemplar OU noise: `sigma=0.22`, `tau=0.10 s`
- Ramp from baseline to endpoint: `1 - exp(-t/0.18)`

Figure 4's DDM uses its own time grid (`T_DDM_MAX = 2.0 s`) since the bound
crossings extend past the 1 s stimulus window.

## SI formulations

Two forms appear:

- **Bounded** (`si_normalized` in framework_common.py):
  `s_c = (1 + cos θ_c) / 2`, then `SI = (s_R - s_L) / (s_R + s_L)`.
  Used in fig 3.
- **Linear / log-LR** (`cos_R - cos_L` directly):
  Used in fig 4 and 6 as the SPRT-optimal increment fed into the accumulator.

Switching between them is a matter of replacing the SI function call in
the relevant figure script.
