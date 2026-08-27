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

# /scratch is per-node, so a cache built there is invisible from anywhere else.
# That is normally a bug, hence the refusal. It is CORRECT when the array is
# pinned to one node and the result is handed to `dataset create` from that node
# -- the registry then replicates it and the local copy stops mattering. Requires
# both CACHE_ALLOW_SCRATCH=1 and CACHE_NODELIST, so it cannot happen by accident.
case "$OUT_ROOT" in
    /scratch/*)
        if [ "${CACHE_ALLOW_SCRATCH:-0}" != "1" ] || [ -z "${CACHE_NODELIST:-}" ]; then
            echo "refusing: /scratch is per-node. Set CACHE_ALLOW_SCRATCH=1 AND" >&2
            echo "CACHE_NODELIST=<node> so every task lands on the same disk." >&2
            exit 1
        fi
        echo "[submit] /scratch build pinned to ${CACHE_NODELIST}"
        ;;
esac
N=$(grep -c . "$LIST")
mkdir -p "$LOG_DIR" "$OUT_ROOT"
# Tasks read the list from LIST_FOR_TASKS, which must be visible on the node that
# RUNS them -- not on the node that submits. Copying it into $OUT_ROOT breaks the
# moment $OUT_ROOT is on /scratch: the copy lands on the submitting node's local
# disk and every task fails with "sed: can't read .../scene_list.txt". /home is
# zone-shared, so the original path always resolves.
LIST_ABS="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
case "$OUT_ROOT" in
    /scratch/*) LIST_FOR_TASKS="$LIST_ABS" ;;
    *) cp "$LIST" "$OUT_ROOT/scene_list.txt"; LIST_FOR_TASKS="$OUT_ROOT/scene_list.txt" ;;
esac

sbatch --parsable \
    --job-name="tapcache" \
    --array="0-$((N-1))%${CONCURRENT}" \
    --partition=batch \
    --constraint="$ZONE" \
    ${CACHE_NODELIST:+--nodelist="$CACHE_NODELIST"} \
    --gpus="$GPU" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --chdir="$REPO_DIR" \
    --output="$LOG_DIR/tapcache_%A_%a.out" \
    scripts/memory/cache_array_job.sh "$LIST_FOR_TASKS" "$ROOT" "$CLIP_LEN" "$STRIDE" "$OUT_ROOT"
