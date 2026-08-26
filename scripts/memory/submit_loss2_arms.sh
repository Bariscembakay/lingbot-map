#!/usr/bin/env bash
# Loss 1 + Loss 2, four points on the four sweep axes.
#
# The axes, all composable and all owned by `lingbot_map/memory/streams.py` so they
# cannot drift apart:
#
#   loss2        recall (teacher's cached tokens) | hindsight (GT depth)
#   query_mode   per_half (8 reads) | per_tap (4) | single (1, then a linear map)
#   write_input  tap23_half | all_second | all_full
#   write_mode   disjoint | overlap
#
# Two composition rules the code enforces rather than trusts:
#   * recall supervises ONLY streams the write ingested -- the state cannot recall
#     what it was never given, and training it to try teaches the read to invent;
#   * the raymap query path has its own residual gate, open at init. Zero-init is
#     for the token path's attributability; sharing it here would make
#     d(loss2)/d(state) exactly zero and leave the write with no gradient at all.
#
# Usage: bash scripts/memory/submit_loss2_arms.sh <cache_root> <out_root> [dep_jobid]
set -euo pipefail

CACHE="$1"; OUT_ROOT="$2"; DEP="${3:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# msp3 by default: that is where the cache lives and where the free H200s are.
ZONE="${ARM_ZONE:-zone-msp3}"
GPU="${ARM_GPU:-h200:1}"
TIME="${ARM_TIME:-12:00:00}"
CPUS="${ARM_CPUS:-16}"
MEM="${ARM_MEM:-96G}"
HEADS="${ARM_HEADS:-/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt}"
LOG_DIR="${ARM_LOG_DIR:-/group/compact-3dmem/campaigns/_joblogs}"
UPDATES="${ARM_UPDATES:-8000}"
EXTRA="${ARM_EXTRA:-}"

DEP_ARG=()
[ -n "$DEP" ] && DEP_ARG=(--dependency="afterok:$DEP")
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Arm names carry every axis, because those are exactly what makes numbers
# incomparable between runs.
#   name | refine_taps | flags
ARMS=(
  "hs_dj_qsingle_wtap23|0,1,2,3|--loss2 hindsight --write-mode disjoint --query-mode single  --write-input tap23_half"
  "rc_dj_qsingle_wtap23|0,1,2,3|--loss2 recall    --write-mode disjoint --query-mode single  --write-input tap23_half"
  "hs_ov_qsingle_wtap23|0,1,2,3|--loss2 hindsight --write-mode overlap  --query-mode single  --write-input tap23_half"
  "rc_dj_qperhalf_wall8|0,1,2,3|--loss2 recall    --write-mode disjoint --query-mode per_half --write-input all_full"
)

for spec in "${ARMS[@]}"; do
    IFS='|' read -r name taps flags <<< "$spec"
    sbatch --parsable \
        --job-name="l2_${name}" \
        --partition=batch \
        --constraint="$ZONE" \
        --gpus="$GPU" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --chdir="$REPO_DIR" \
        --output="$LOG_DIR/l2_${name}.out" \
        "${DEP_ARG[@]}" \
        scripts/memory/arm_job.sh "$CACHE" "$OUT_ROOT/$name" "$name" "$taps" "$HEADS" \
        "--with-camera --max-updates $UPDATES $flags $EXTRA"
done
