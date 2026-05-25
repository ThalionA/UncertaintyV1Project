#!/usr/bin/env bash
# run_all_scaling.sh -- population-scaling sweep across targets.
#
# Targets run SEQUENTIALLY, in the order given (default: Q d L choice
# stim_cat stim_kernel). Within each target the 6 mice run in PARALLEL
# -- one OS process per mouse, the GPU multiplexing them. The point of
# this layout (vs one process per target) is that a target finishes
# COMPLETELY before the next starts, so its full per-mouse and
# cross-mouse visualisations are available while later targets run.
# Total wall-clock is unchanged -- the GPU is the bottleneck either way.
#
# Per target:
#   1. fan the 6 mice across 6 `--no-aggregate` worker processes; each
#      writes only its own per-mouse CSV (race-free).
#   2. a reconcile pass (all mice, every per-mouse CSV now cached -> no
#      training) writes that target's combined + aggregate CSVs and the
#      provenance JSON.
#
# Usage (from anywhere):
#   PY=~/cluster-env/.venv/bin/python bash run_all_scaling.sh [TARGET ...]
#
#   [TARGET ...] positional target subset / order   (default: Q d L choice stim_cat stim_kernel)
#   PY          python interpreter                  (default: `python`)
#   MIN_EPOCHS  num_epochs floor (preset if higher)  (default: 100)
#   N_DRAWS     random subsets drawn per size        (default: 5)
#   STEP        neuron-grid step                     (default: 10)
#   TARGETS     env-var alternative to positionals   (space-separated)
#
# Logs: neuron_scaling_logs/<target>_mouse<N>.log per worker,
#       neuron_scaling_logs/<target>_reconcile.log per reconcile.
# Run inside tmux so it survives an ssh disconnect.
#
# NOTE: neuron_scaling.py is resumable -- an existing per-mouse CSV is
# reused. For a from-scratch run, delete results/neuron_scaling/ first.

set -u

cd "$(dirname "$0")"                       # -> nn_decoder/
PY="${PY:-python}"
MIN_EPOCHS="${MIN_EPOCHS:-100}"
N_DRAWS="${N_DRAWS:-5}"
STEP="${STEP:-10}"
LOGDIR="neuron_scaling_logs"
mkdir -p "$LOGDIR"

# Target order: positional args win; else the TARGETS env var; else the
# default order (Q, d, L first -- the priority targets -- then the rest).
if [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    read -ra TARGETS <<< "${TARGETS:-Q d L choice stim_cat stim_kernel}"
fi
MICE=(0 1 2 3 4 5)

echo "Scaling sweep | python=$PY | min_epochs=$MIN_EPOCHS | n_draws=$N_DRAWS | step=$STEP"
echo "Target order (sequential): ${TARGETS[*]}"
fail=0

for t in "${TARGETS[@]}"; do
    echo
    echo "######## TARGET $t : ${#MICE[@]} mice in parallel ########"
    pids=()
    for mid in "${MICE[@]}"; do
        log="$LOGDIR/${t}_mouse${mid}.log"
        "$PY" neuron_scaling.py --target "$t" --mouse-ids "$mid" \
            --no-aggregate --min-epochs "$MIN_EPOCHS" \
            --n-draws "$N_DRAWS" --step "$STEP" > "$log" 2>&1 &
        pids+=("$!")
        echo "  $t mouse $mid -> pid $!  (log: $log)"
        sleep 20                           # stagger the .mat-load RAM spike
    done
    for i in "${!MICE[@]}"; do
        if wait "${pids[$i]}"; then
            echo "  done  $t mouse ${MICE[$i]}"
        else
            echo "  FAIL  $t mouse ${MICE[$i]}  -- see $LOGDIR/${t}_mouse${MICE[$i]}.log"
            fail=1
        fi
    done
    # Reconcile: every per-mouse CSV now cached -> no training, just this
    # target's combined + aggregate CSVs and provenance JSON.
    rlog="$LOGDIR/${t}_reconcile.log"
    if "$PY" neuron_scaling.py --target "$t" --min-epochs "$MIN_EPOCHS" \
            --n-draws "$N_DRAWS" --step "$STEP" > "$rlog" 2>&1; then
        echo "######## TARGET $t COMPLETE -- combined + aggregate ready ########"
    else
        echo "  FAIL  $t reconcile -- see $rlog"
        fail=1
    fi
done

echo
echo "ALL SCALING TARGETS FINISHED (fail=$fail)"
exit "$fail"
