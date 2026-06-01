#!/usr/bin/env bash
# run_loss_sweep_per_mouse.sh -- launch the Q-target loss sweep one
# loss at a time, parallelising across the 6 mice within each loss.
#
# Per the 2026-05-27 meeting plan ([[2026-05-27-Uncertainty-Meeting]]):
#   - one loss at a time (serial outer loop)
#   - 6 mice in parallel within a loss (inner background launch)
#   - idempotent: skip a loss whose canonical <split>.mat is already on
#     disk, skip an individual mouse whose shard is already on disk
#
# Usage (from anywhere; the script cd's into nn_decoder/):
#   PY=~/cluster-env/.venv/bin/python bash run_loss_sweep_per_mouse.sh
#   PY=~/cluster-env/.venv/bin/python RUN_NAME=my_sweep \
#     bash run_loss_sweep_per_mouse.sh PCA MSE
#
#   PY            python interpreter           (default: `python`)
#   RUN_NAME      results subdirectory         (default: loss_sweep_peakiness_2026_05_27)
#   NUM_EPOCHS    epochs per fit               (default: 100)
#   SNAPSHOT      weight-snapshot cadence      (default: 10 epochs)
#   HIDDEN_SIZES  override hidden layer sizes  (default: empty = per-target Optuna preset)
#                 e.g. HIDDEN_SIZES="10" for a [10] layer, "16 16" for [16,16]
#   VAL_FRAC      fraction of train carved off as val (default: 0 = off)
#   MICE          space-separated mouse ids    (default: 0 1 2 3 4 5)
#   STAGGER       seconds between mouse launches (default: 20 — spreads
#                 the 558 MB VR-export load RAM spike)
#   [LOSSES…]     positional args = loss subset (default: all 6)
#
# Run inside tmux so it survives an ssh disconnect:
#   tmux new -s lossweep
#   PY=~/cluster-env/.venv/bin/python bash run_loss_sweep_per_mouse.sh
#   # Ctrl-B D to detach; `tmux attach -t lossweep` to reattach.
#
# Resumption: if a loss crashes part-way through, just re-run the same
# command. Completed mice (those with shards on disk) are skipped;
# missing mice are refit. The merge step runs only when every mouse
# for that loss has produced a shard.

set -u

cd "$(dirname "$0")"                       # -> nn_decoder/
PY="${PY:-python}"
RUN_NAME="${RUN_NAME:-loss_sweep_peakiness_2026_05_27}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
SNAPSHOT="${SNAPSHOT:-10}"
STAGGER="${STAGGER:-20}"
HIDDEN_SIZES="${HIDDEN_SIZES:-}"
VAL_FRAC="${VAL_FRAC:-0}"
SPLIT="stratified_balanced"

# Build extra CLI flags from the env-var overrides. Empty when the
# override is unset, so the default Config preset takes over.
EXTRA_FLAGS=()
if [ -n "$HIDDEN_SIZES" ]; then
    # Pass each size as a separate token after --hidden-sizes.
    read -ra HS <<< "$HIDDEN_SIZES"
    EXTRA_FLAGS+=(--hidden-sizes "${HS[@]}")
fi
# Always pass --val-frac (0 is a valid value that the script handles).
EXTRA_FLAGS+=(--val-frac "$VAL_FRAC")

# Losses: positional args win; else the default set (MSE dropped on
# 2026-05-27 — see run_loss_sweep.py for context).
if [ "$#" -gt 0 ]; then
    LOSSES=("$@")
else
    LOSSES=(PCA CE JS KL Wasserstein)
fi

read -ra MICE <<< "${MICE:-0 1 2 3 4 5}"

echo "Q-target loss sweep (one loss at a time, mice in parallel)"
echo "  run_name : $RUN_NAME"
echo "  losses   : ${LOSSES[*]}"
echo "  mice     : ${MICE[*]}"
echo "  epochs   : $NUM_EPOCHS"
echo "  stagger  : ${STAGGER}s between mouse launches"
echo "  python   : $PY"
echo

# Helper: compute the slug for a (target, loss, window, bin, basis).
# Calls into training.config so the wrapper stays in lock-step with
# Config.slug() — no hard-coded path-template duplication here.
slug_for_loss() {
    local loss="$1"
    "$PY" -c "import sys; sys.path.insert(0, '.'); \
from training.config import default_config_for_target; \
print(default_config_for_target('Q', bin_size_ms=100, loss_func='$loss').slug())"
}

overall_fail=0
for loss in "${LOSSES[@]}"; do
    slug=$(slug_for_loss "$loss")
    slug_dir="results/${RUN_NAME}/${slug}"
    final_mat="${slug_dir}/${SPLIT}.mat"

    if [[ -f "$final_mat" ]]; then
        echo "[skip] $loss: ${final_mat} already exists"
        continue
    fi

    echo "==========================================================="
    echo "Loss: $loss  (slug=${slug})"
    echo "==========================================================="
    logdir="${slug_dir}/logs"
    mkdir -p "$logdir"

    pids=()
    for mid in "${MICE[@]}"; do
        shard="${slug_dir}/shards/${SPLIT}__mouse_${mid}.mat"
        if [[ -f "$shard" ]]; then
            echo "  [skip] mouse $mid: shard already exists"
            continue
        fi
        log="${logdir}/mouse_${mid}.log"
        "$PY" run_loss_sweep.py \
            --run-name "$RUN_NAME" \
            --losses "$loss" \
            --num-epochs "$NUM_EPOCHS" \
            --snapshot-every "$SNAPSHOT" \
            --mouse-id "$mid" \
            "${EXTRA_FLAGS[@]}" \
            > "$log" 2>&1 &
        pids+=("$!:$mid")
        echo "  mouse $mid -> pid $! (log: $log)"
        sleep "$STAGGER"                  # spread VR-export load RAM spike
    done

    # Wait on each launched process.
    loss_fail=0
    for entry in "${pids[@]}"; do
        pid="${entry%%:*}"; mid="${entry##*:}"
        if wait "$pid"; then
            echo "  DONE mouse $mid"
        else
            echo "  FAIL mouse $mid  (see ${logdir}/mouse_${mid}.log)"
            loss_fail=1
        fi
    done

    if [[ $loss_fail -eq 0 ]]; then
        echo "Merging shards for $loss..."
        "$PY" -c "import sys; sys.path.insert(0, '.'); \
from pathlib import Path; \
from training.run import merge_shards; \
merge_shards(Path('${slug_dir}'), '${SPLIT}')"
        if [[ -f "$final_mat" ]]; then
            echo "  $final_mat written"
        else
            echo "  [!] merge produced no .mat (no shards on disk?)"
            overall_fail=1
        fi
    else
        echo "Skipping merge for $loss: re-run to refill the missing mice."
        overall_fail=1
    fi
done

echo
echo "ALL LOSSES FINISHED (fail=$overall_fail)"
exit "$overall_fail"
