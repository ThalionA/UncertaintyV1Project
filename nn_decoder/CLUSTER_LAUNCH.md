# CLUSTER_LAUNCH — sending decoder jobs to gpu1

Canonical, copy-pasteable recipe for launching `run_loss_comparison.py` sweeps on the
remote GPU. Companion to `DATA_MAP.md` (what the outputs are) and `wiki/Cluster_Commands.md`
(tmux basics). **Update this if the paths/flags change.**

## Fixed coordinates
| | |
|---|---|
| Local repo (this Mac) | `~/Desktop/Experiments/UncertaintyV1` |
| Remote repo (gpu1) | `~/UncertaintyV1` |
| Remote python (the env) | `~/cluster-env/.venv/bin/python` |
| ssh alias | `gpu1` (`NVIDIA RTX A5000`) |
| Runner | `nn_decoder/run_loss_comparison.py` (**run from `nn_decoder/`**) |
| Outputs | `~/UncertaintyV1/nn_decoder/results/<run_name>/<cell>/` |

**Gotchas (have bitten):**
- `$PY` is **not** a global on gpu1 — define it per session: `PY=~/cluster-env/.venv/bin/python` (or inline the full path). A bare `$PY -u …` → `-u: command not found`.
- **`cd ~/UncertaintyV1/nn_decoder` first** — the runner imports sibling modules and writes `results/` relative to cwd.
- **The agent CAN `ssh`/`rsync` gpu1 directly** (verified 2026-08-12: connect, rsync up, `tmux new -d`, launch, poll the log, read results back). This line previously said the harness blocked it and that Theo had to run every command by hand — that has been false since the allow-rule landed on 2026-08-08. Still ask before launching: a run is Theo's GPU time. Env on the box is `~/cluster-env/.venv/bin/python` (memory `cluster-launch-access`).
- **The remote repo drifts far behind local** — it was 10+ commits behind on 2026-08-12 (`c1de343` vs local `98b35fd`) while its *files* were partly newer, because past sessions rsync'd source without committing. So `git log` on gpu1 does **not** tell you what code is about to run. Verify the actual thing you changed (`grep` the symbol on the remote file), and rsync with `--backup --suffix=.pre_<job>_<date>` so an overwrite is recoverable. Check `git status` on gpu1 before syncing and skip any file with uncommitted remote edits.
- **No `timeout` on macOS** — `timeout 25 ssh …` fails with `command not found`; use `ssh -o ConnectTimeout=N` instead.
- Local test suite segfaults under multithreaded BLAS → `OMP_NUM_THREADS=1` (see `GOTCHAS.md`).

**Launch discipline (cheap, has already paid for itself):**
1. `--dry-run` **on the remote** first — catches import skew from the drift above before any GPU time.
2. `--smoke` next (1 mouse, 2 epochs; writes to a separate `<run>_smoke` tree, so it cannot pollute results).
3. Then verify the smoke output actually shows the property you changed, from the saved `.mat` — not from the config you *intended*. On 2026-08-12 this is what proved the new `rr8` cells were genuinely rank-8 rather than a silently-substituted ReLU net. Note weights live under `results[mouse]['Weights'][arch]['W_in'|'W_out']`, **not** under `Dist`.
4. Only then launch the full run under `tmux new -d -s <job>` with `| tee <job>.log`.

## The four steps

### 1. Push code (up) — from the Mac, repo root
```bash
cd ~/Desktop/Experiments/UncertaintyV1
rsync -avz nn_decoder/run_loss_comparison.py nn_decoder/nn_classifier.py \
          nn_decoder/run_experiment.py gpu1:~/UncertaintyV1/nn_decoder/
rsync -avz nn_decoder/training/config.py gpu1:~/UncertaintyV1/nn_decoder/training/   # note the training/ subdir
```
(Push only the files you changed; `training/config.py` must keep its subdir path.)

### 2. Launch — on gpu1, inside tmux
```bash
ssh gpu1
tmux new -s <jobname>                 # detach later: Ctrl-b then d  (NOT Ctrl-c)
cd ~/UncertaintyV1/nn_decoder
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/cluster-env/.venv/bin/python
$PY -u run_loss_comparison.py <flags> 2>&1 | tee <jobname>.log
```

### 3. Monitor — from the Mac (no attach needed)
```bash
ssh gpu1 "tail -n 40 ~/UncertaintyV1/nn_decoder/<jobname>.log"
ssh gpu1 "ls -d ~/UncertaintyV1/nn_decoder/results/<run_name>*"               # which cells landed
ssh gpu1 "grep -iE 'error|traceback|cuda|killed' ~/UncertaintyV1/nn_decoder/<jobname>.log | tail"
# interactive: ssh gpu1; tmux attach -t <jobname>   (Ctrl-b d to leave it running)
```

### 4. Pull results (down) — from the Mac, repo root
```bash
rsync -avz 'gpu1:~/UncertaintyV1/nn_decoder/results/<run_name>*' nn_decoder/results/
```
Figures are generated **locally** from the pulled `.mat`s (they're gitignored; don't sync them up).

## `run_loss_comparison.py` flags → isolated run-name suffix
| flag | effect | run_name suffix |
|---|---|---|
| `--targets Q L` `--bin-sizes-ms 50 100` `--windows half full` `--splits …` | the grid | — |
| `--losses PCA CE KL JS Wasserstein` | which losses (matched) | — |
| `--patience N` | early stopping (N=0 → fixed `--num-epochs`) | — |
| `--hidden-sizes 4 8 16 32 64` | width ablation | `_h<H>` |
| `--entropy-lambdas 0 0.001 0.003 0.01 0.03 0.1` | λ_H sweep (temporal only) | `_entlam<λ>` |
| `--dropout 0.25` | dropout reg (pair `--patience 0` to isolate) | `_drop<p>` |
| `--shape-lambda 0.1` / `--evar-alpha 0.3` / `--flat-evar` | PCA-loss width controls | `_shape<λ>` / `_alpha<α>` / `_flatevar` |
(dots in suffixes → `p`, e.g. `_entlam0p003`. Defaults reproduce `loss_comparison_v1`.)

## Worked examples
```bash
# λ_H sweep (done 2026-06-14)
$PY -u run_loss_comparison.py --run-name lambdaH_sweep \
    --entropy-lambdas 0 0.001 0.003 0.01 0.03 0.1 \
    --targets Q --bin-sizes-ms 100 --windows half --splits stratified_balanced 2>&1 | tee lambdaH_sweep.log
# -> results/lambdaH_sweep_entlam0 … _entlam0p1 ;  pull: 'results/lambdaH_sweep_entlam*'

# dropout vs early stopping (dropout-only + no-reg reference; early-stop-only = loss_comparison_v1)
for p in 0.1 0.25 0.5; do
  $PY -u run_loss_comparison.py --run-name dropreg --dropout $p --patience 0 \
      --targets Q --bin-sizes-ms 100 --windows half --splits stratified_balanced 2>&1 | tee dropreg_$p.log
done
$PY -u run_loss_comparison.py --run-name noreg --patience 0 \
    --targets Q --bin-sizes-ms 100 --windows half --splits stratified_balanced 2>&1 | tee noreg.log
# pull: 'results/dropreg_drop*' and 'results/noreg'
```
