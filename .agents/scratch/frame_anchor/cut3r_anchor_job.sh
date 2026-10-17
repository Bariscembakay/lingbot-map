#!/usr/bin/env bash
# Frame-anchor diagnostic: stream scenes through published CUT3R, ship per-frame
# records to the campaign. See cut3r_anchor_dump.py.
#
# Usage: sbatch [...] cut3r_anchor_job.sh <benchmark> <scene> [<scene> ...]
#   SMOKE=1      truncate to 40 frames, keep output on this node's /scratch only
#   CHECK=1      run check_stream_equivalence.py first and abort on FAIL
#   ORDER=palindrome|forward|repeat   ARM=<arm dir name>
#   EXTRA_ARGS="..."  passed to cut3r_anchor_dump.py (e.g. --start/--end/--repeat)
#   SCENE_TAG=_f214-484  appended to the scene's output dir name
set -euo pipefail

BENCH="$1"; shift
REPO_DIR="$HOME/lingbot-map"
HERE="$REPO_DIR/.agents/scratch/frame_anchor"
ORDER="${ORDER:-palindrome}"
ARM="${ARM:-cut3r_pub_${ORDER}_s1}"
CAMPAIGN=/group/compact-3dmem/campaigns/frame_anchor
WORK="/scratch/$USER/${SLURM_JOB_ID:-local}/frame_anchor"

# The env build can take ~15 min with the GPU idle, which is what the idle-GPU
# reaper looks for. Hold the GPU with whatever torch env this node already has.
KA_PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"
KA_PID=""
if [ -x "$KA_PY" ] && [ ! -d "${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r" ]; then
    "$KA_PY" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
    KA_PID=$!
fi
trap '[ -n "$KA_PID" ] && kill $KA_PID 2>/dev/null || true' EXIT

source "$REPO_DIR/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"
[ -n "$KA_PID" ] && { kill "$KA_PID" 2>/dev/null || true; KA_PID=""; }
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"

case "$BENCH" in
    nrgbd)          dataset pull NRGBD >/dev/null; ROOT=/data/NRGBD ;;
    kitti)          dataset pull kitti_odometry >/dev/null; ROOT=/data/kitti_odometry ;;
    kitti_depthval) ROOT=/group/compact-3dmem/datasets/kitti_depth ;;
    *) echo "unknown benchmark $BENCH" >&2; exit 2 ;;
esac

echo "[job] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) bench=$BENCH arm=$ARM scenes=$*"

if [ "${CHECK:-0}" = 1 ]; then
    "$PY" "$HERE/check_stream_equivalence.py"
fi

EXTRA=()
[ "${SMOKE:-0}" = 1 ] && EXTRA+=(--max-frames 40)

for SCENE in "$@"; do
    OUT="$WORK/$ARM/$BENCH/$SCENE${SCENE_TAG:-}"
    # shellcheck disable=SC2086
    "$PY" "$HERE/cut3r_anchor_dump.py" --benchmark "$BENCH" --scene "$SCENE" --root "$ROOT" \
        --order "$ORDER" --out "$OUT" "${EXTRA[@]}" ${EXTRA_ARGS:-}
    if [ "${SMOKE:-0}" != 1 ]; then
        DEST="$CAMPAIGN/$ARM/$BENCH/$SCENE${SCENE_TAG:-}"
        mkdir -p "$DEST"
        rsync -a "$OUT/" "$DEST/"
        echo "[job] shipped $OUT -> $DEST"
    fi
done
