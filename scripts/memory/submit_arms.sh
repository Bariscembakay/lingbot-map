#!/usr/bin/env bash
# Submit the training arms, one GPU each.
#
# Four arms, not two. The frozen-state controls are not optional: the read is
# queried by the current frame's own tokens, so an arm can improve with the state
# contributing nothing -- a residual adapter correcting the frozen head's bias
# looks exactly like success. Beating frozen lingbot-map proves nothing; beating
# the frozen-state arm does.
#
# Usage: bash scripts/memory/submit_arms.sh <cache_root> <out_root> [dependency_jobid]
set -euo pipefail

CACHE="$1"; OUT_ROOT="$2"; DEP="${3:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ZONE="${ARM_ZONE:-zone-sof1}"
GPU="${ARM_GPU:-h200:1}"
TIME="${ARM_TIME:-12:00:00}"
CPUS="${ARM_CPUS:-16}"
MEM="${ARM_MEM:-96G}"
HEADS="${ARM_HEADS:-/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt}"
LOG_DIR="${ARM_LOG_DIR:-$REPO_DIR/.agents/scratch/memory_logs}"
EXTRA="${ARM_EXTRA:-}"

DEP_ARG=()
[ -n "$DEP" ] && DEP_ARG=(--dependency="afterok:$DEP")
mkdir -p "$LOG_DIR" "$OUT_ROOT"

#   name              refine_taps  extra flags
ARMS=(
  "A_full|0,1,2,3|--with-camera"
  "A_frozenstate|0,1,2,3|--with-camera --frozen-state"
  "B_pose|3|--with-camera"
  "B_frozenstate|3|--with-camera --frozen-state"
)

for spec in "${ARMS[@]}"; do
    IFS='|' read -r name taps flags <<< "$spec"
    sbatch --parsable \
        --job-name="mem_${name}" \
        --partition=batch \
        --constraint="$ZONE" \
        --gpus="$GPU" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --chdir="$REPO_DIR" \
        --output="$LOG_DIR/arm_${name}.out" \
        "${DEP_ARG[@]}" \
        scripts/memory/arm_job.sh "$CACHE" "$OUT_ROOT/$name" "$name" "$taps" "$HEADS" "$flags $EXTRA"
done
