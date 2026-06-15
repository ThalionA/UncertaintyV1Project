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
- The **agent cannot `ssh`/`rsync` gpu1** (harness blocks it) — Theo runs these; the agent generates the block (memory `cluster-launch-access`).
- Local test suite segfaults under multithreaded BLAS → `OMP_NUM_THREADS=1` (see `GOTCHAS.md`).

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
