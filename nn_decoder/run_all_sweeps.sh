#!/usr/bin/env bash
# run_all_sweeps.sh -- launch every per-target Optuna sweep concurrently.
#
# optuna_per_target.py writes a SEPARATE SQLite study per target, so the
# six sweeps are fully independent. Running them as six OS processes lets
# the one GPU multiplex their (individually light, latency-bound)
# workloads -- this is the parallelism, with no change to the sweep code.
#
# Usage (from anywhere):
#   PY=~/cluster-env/.venv/bin/python bash run_all_sweeps.sh [TARGET ...]
#
#   [TARGET ...] positional target subset          (default: all six)
#   PY          python interpreter to use          (default: `python`)
#   N_TRIALS    trials per target                  (default: 100)
#   PCA_BASIS   PCA loss-basis for PCA-loss targets (default: all_trials)
#   TARGETS     env-var alternative to positionals  (space-separated)
#
# Examples:
#   # re-sweep only the PCA-loss targets under the all_trials basis:
#   PCA_BASIS=all_trials bash run_all_sweeps.sh Q L stim_kernel
#
# Each sweep logs to optuna_studies/logs/<target>_<pca_basis>.log. The
# script blocks until all targets finish -- run it inside tmux so it
# survives an ssh disconnect.
#
# NOTE: optuna_per_target.py resumes an existing study (load_if_exists),
# but the study name now includes the pca_basis, so runs under different
# bases never collide. For a from-scratch run of a given (target, basis),
# delete optuna_studies/<target>_*_<basis>.db first.

set -u

cd "$(dirname "$0")"                       # -> nn_decoder/
PY="${PY:-python}"
N_TRIALS="${N_TRIALS:-100}"
PCA_BASIS="${PCA_BASIS:-all_trials}"
LOGDIR="optuna_studies/logs"
mkdir -p "$LOGDIR"

# Targets: positional args win; else the TARGETS env var (space-
# separated); else all six.
if [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    read -ra TARGETS <<< "${TARGETS:-Q L d choice stim_cat stim_kernel}"
fi

echo "Launching ${#TARGETS[@]} sweeps | python=$PY | n_trials=$N_TRIALS | pca_basis=$PCA_BASIS"
pids=()
for t in "${TARGETS[@]}"; do
    log="$LOGDIR/${t}_${PCA_BASIS}.log"
    "$PY" optuna_per_target.py --target "$t" --n-trials "$N_TRIALS" \
        --pca-basis "$PCA_BASIS" > "$log" 2>&1 &
    pids+=("$!")
    echo "  $t -> pid $!  (log: $log)"
    sleep 20                               # stagger: spread the .mat-load RAM spike
done

echo "All launched. Waiting for completion..."
fail=0
for i in "${!TARGETS[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  DONE  ${TARGETS[$i]}"
    else
        echo "  FAIL  ${TARGETS[$i]}  -- see $LOGDIR/${TARGETS[$i]}_${PCA_BASIS}.log"
        fail=1
    fi
done

echo "ALL SWEEPS FINISHED (fail=$fail)"
echo "Winning presets: grep -A12 'Paste into' $LOGDIR/*_${PCA_BASIS}.log"
exit "$fail"
