# Handoff — KL/JS sweep plotting & next steps

_Date: 2026-06-02. Branch: `main` (HEAD `6f5ce80`)._

## TL;DR

Plotting for the KL/JS entropy-lambda sweep is built and tested. **Two of the
four latest requests work on the data you already have; two need a re-run on
the cluster with history tracking on** (the first sweep didn't capture per-epoch
curves or weight snapshots).

| Request | Status | Script |
|---|---|---|
| Q (perceptual) and L (likelihood) **separately** | ✅ ready | `plot_kl_js_sweep.py` (one matched fig per target) |
| Per-bin posteriors for **KL and JS separately** | ✅ ready | `plot_kl_js_sweep.py` fig3 (per-loss columns) |
| Per-bin posteriors **across lambda**, matched trials | ✅ ready | `plot_kl_js_sweep.py` fig6 |
| spat (PPC) vs temp (SBC) side by side | ✅ ready | `plot_kl_js_sweep.py` fig5 |
| **Train vs val loss across epochs + early-stop point** | ⚠ needs tracked re-run | `plot_kl_js_training.py` fig A |
| **Weight evolution** | ⚠ needs tracked re-run | `plot_kl_js_training.py` fig B |

## Why training/weight plots need a re-run

The original sweep ran with `track_training_history=False` and
`weight_snapshot_every=0` (the Config defaults). So the per-epoch train/val
curves and weight snapshots **were never saved** — they are not recoverable from
the existing `.mat`/`.pt` files. `plot_kl_js_training.py` detects this and prints
a message rather than emitting empty figures.

## What to do

### 1. Plots that work NOW (on the results already rsync'd)

```bash
git checkout main && git pull
cd nn_decoder
python plot_kl_js_sweep.py            # both archs in one call
# -> figures/kl_js_sweep/kl_js_entropy_sweep_v1/
```

Figures produced (overlapping/superseded plots have been pruned):
- `2_sweep_over_knobs_{spat,temp}.png` — decoded entropy vs λ, per bin/window.
- `3_matched_examples_<target>_lam<λ>.png` — **one figure per (target, λ)**;
  columns `[spat | temp time-avg | per-bin KL | per-bin JS]`, matched trials,
  **y-lim matched across each row**.
- `4_fit_loss.png` — held-out fit-loss; **spat (solid) and temp (dashed) in the
  same subplot** per target, shared y-axis.
- `5_peakiness_spat_vs_temp.png` — decoded peakiness vs target; rows =
  entropy / max-prob, columns = spat (PPC) / temp (SBC). (Replaces the old
  per-arch `1_peakiness_vs_targets_*`, whose max-prob panel is folded in here.)
- `6_perbin_vs_lambda_<target>_<loss>.png` — **per-bin posteriors across λ**
  (1e-3/3e-3/1e-2) for matched trials; faint = time-bins, bold = mean, y-lim
  matched per row, per-bin H̄ + avg-posterior H annotated. One per (target,loss).
- `7_avg_perbin_entropy_vs_lambda.png` — **mean per-bin posterior entropy vs λ**,
  one line per (target, loss); the summary of how λ sharpens per-bin posteriors.
- `summary.csv` — every config × arch.

### 2. Get the training-curve + weight-evolution plots

Re-run the sweep (or a subset) on the cluster WITH tracking. Recommend a small
subset first since snapshots add disk + the curves are most interesting at
100ms/half:

```bash
# on gpu1, in tmux
cd ~/UncertaintyV1/nn_decoder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/cluster-env/.venv/bin/python

# subset with tracking (one new run-name so it doesn't collide with v1):
$PY -u run_kl_js_entropy_sweep.py \
    --run-name kl_js_entropy_sweep_tracked \
    --bin-sizes-ms 100 --windows half \
    --splits stratified_balanced \
    --track-history --snapshot-every 10 \
    2>&1 | tee kljs_tracked.log
```

That = 2 targets × 2 losses × 3 λ × 1 bin × 1 window × 6 mice = 36 fits, each
writing per-epoch curves + weight snapshots into
`checkpoints/mouse_<id>_<split>.pt`.

Then rsync down and plot:
```bash
# laptop
rsync -avz gpu1:~/UncertaintyV1/nn_decoder/results/kl_js_entropy_sweep_tracked/ \
  nn_decoder/results/kl_js_entropy_sweep_tracked/
cd nn_decoder
python plot_kl_js_training.py --run-name kl_js_entropy_sweep_tracked --arch temp --mouse 0
python plot_kl_js_training.py --run-name kl_js_entropy_sweep_tracked --arch spat --mouse 0
# -> figures/kl_js_sweep/kl_js_entropy_sweep_tracked/training/
#    A_training_curves_<arch>_mouse0.png  (★=best epoch, vertical line=early stop)
#    B_weight_evolution_<arch>_mouse0.png (per-parameter L2 norm vs epoch)
```

## Code changes in this handoff (all on `main`, `6f5ce80`)

- `nn_classifier.py` — `fit_model` now records `early_stopped_epoch`,
  `best_epoch`, `epoch_cap`, `patience` into the history dict (scalars). No
  behavioural change to training; `tests/test_early_stopping.py` still passes.
- `run_kl_js_entropy_sweep.py` — `--track-history` / `--snapshot-every` flags
  (default off, so the production schedule is unchanged).
- `plot_kl_js_sweep.py` — Q/L-separate matched figures, per-loss per-bin
  columns, new fig6 (per-bin vs λ). Loads both archs + `decoded_samp`.
- `plot_kl_js_training.py` — NEW; training curves + weight evolution from `.pt`.

## Gotchas / honest caveats

- **Per-bin view is temp/SBC only.** PPC integrates over time before softmax,
  so `decoded_samp` is empty for spat — there are no per-bin posteriors there.
- **Matched trials are valid** because the test split is seeded identically
  across losses (`test_size=0.5, random_state=42` in `run_experiment.py`), so
  trial index `i` is the same physical trial in KL and JS of one config. This
  was verified in code, not assumed.
- **None of the plotting has run on the real `.mat`/`.pt` yet** — only on
  synthetic fixtures matching the documented structure. If a real file has a
  quirk (e.g. a single-mouse `.mat` collapsing a dict via `simplify_cells`),
  a panel may error; paste the traceback and it's a quick fix.
- `nn_decoder/results/` and `figures/` are gitignored — outputs stay local.
