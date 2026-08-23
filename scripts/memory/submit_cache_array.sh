#!/usr/bin/env bash
# Build the dev-set cache as one throttled job array: one task per (scene, clip).
#
# Throttled to 8 concurrent tasks because cluster etiquette is "no more than 8
# H200 at a time" -- an unthrottled 28-task array would take 28.
#
# Usage: bash scripts/memory/submit_cache_array.sh <scene_list> <out_root> <clip_len>
#   scene_list: one "scene:clip" per line
set -euo pipefail

LIST="$1"; OUT_ROOT="$2"; CLIP_LEN="$3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ZONE="${CACHE_ZONE:-zone-sof1}"
GPU="${CACHE_GPU:-h200:1}"
TIME="${CACHE_TIME:-0:25:00}"
CPUS="${CACHE_CPUS:-16}"
MEM="${CACHE_MEM:-64G}"
STRIDE="${CACHE_STRIDE:-20}"
CONCURRENT="${CACHE_CONCURRENT:-8}"
ROOT="${SCANNETPP_ROOT:-/data/ScanNetpp}"
# Job logs belong in the campaign record, not /home -- /home is
# repositories and scripts only, and campaigns/_joblogs already exists.
LOG_DIR="${CACHE_LOG_DIR:-/group/compact-3dmem/campaigns/_joblogs}"

case "$OUT_ROOT" in
    /scratch/*) echo "refusing: /scratch is per-node, the cache must be zone-shared" >&2; exit 1 ;;
esac
N=$(grep -c . "$LIST")
mkdir -p "$LOG_DIR" "$OUT_ROOT"
cp "$LIST" "$OUT_ROOT/scene_list.txt"

sbatch --parsable \
    --job-name="tapcache" \
    --array="0-$((N-1))%${CONCURRENT}" \
    --partition=batch \
    --constraint="$ZONE" \
    --gpus="$GPU" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --chdir="$REPO_DIR" \
    --output="$LOG_DIR/tapcache_%A_%a.out" \
    scripts/memory/cache_array_job.sh "$OUT_ROOT/scene_list.txt" "$ROOT" "$CLIP_LEN" "$STRIDE" "$OUT_ROOT"
