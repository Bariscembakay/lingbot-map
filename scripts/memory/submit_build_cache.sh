#!/usr/bin/env bash
# Submit teacher-cache builds. One job per (scene, clip).
#
# Usage: bash scripts/memory/submit_build_cache.sh <out_root> <clip_len> <scene:clip> ...
#
# out_root must be on a ZONE-SHARED filesystem (/group or /home), never
# /scratch: /scratch is /dev/md40, a per-node local disk, so a cache written by a
# build job on one node is invisible to a training job on another. Same reason the
# sbatch --output path lives under /home -- Slurm opens it on the compute node.
#
#   real builds:  /group/compact-3dmem/datasets/summary_memory_cache_v1
#   smoke tests:  /group/compact-3dmem/datasets/_smoke  (delete, do not file)
set -euo pipefail

OUT_ROOT="$1"; CLIP_LEN="$2"; shift 2
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ZONE="${CACHE_ZONE:-zone-sof1}"
GPU="${CACHE_GPU:-h200:1}"
TIME="${CACHE_TIME:-0:30:00}"
CPUS="${CACHE_CPUS:-16}"
MEM="${CACHE_MEM:-64G}"
STRIDE="${CACHE_STRIDE:-10}"
ROOT="${SCANNETPP_ROOT:-/data/ScanNetpp}"
PARTITION="${CACHE_PARTITION:-batch}"
# hala's debug and rendering partitions each carry AllowQos=<own name>: a job that
# omits the matching --qos is rejected outright, not merely deprioritised.
if [ -n "${CACHE_QOS:-}" ]; then QOS="$CACHE_QOS"; else
    case "$PARTITION" in
        debug)     QOS=debug ;;
        rendering) QOS=rendering ;;
        *)         QOS="" ;;
    esac
fi
QOS_ARG=(); [ -n "$QOS" ] && QOS_ARG=(--qos="$QOS")
LOG_DIR="${CACHE_LOG_DIR:-$REPO_DIR/.agents/scratch/memory_logs}"

case "$OUT_ROOT" in
    /scratch/*) echo "refusing: out_root on per-node /scratch -- use /group" >&2; exit 1 ;;
esac

mkdir -p "$LOG_DIR" "$OUT_ROOT"

for spec in "$@"; do
    scene="${spec%%:*}"; clip="${spec##*:}"
    name="${scene}_c${clip}"
    sbatch \
        --job-name="tapcache_${name}" \
        --partition="$PARTITION" \
        "${QOS_ARG[@]}" \
        --constraint="$ZONE" \
        --gpus="$GPU" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --chdir="$REPO_DIR" \
        --output="$LOG_DIR/${name}.out" \
        scripts/memory/build_cache_job.sh \
            "$ROOT" "$scene" "$clip" "$CLIP_LEN" "$STRIDE" "$OUT_ROOT/${name}"
done
