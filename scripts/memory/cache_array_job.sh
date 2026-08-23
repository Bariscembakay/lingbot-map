#!/usr/bin/env bash
# One array task: pick the (scene, clip) on line $SLURM_ARRAY_TASK_ID and build it.
set -euo pipefail
LIST="$1"; ROOT="$2"; CLIP_LEN="$3"; STRIDE="$4"; OUT_ROOT="$5"

# list lines are "scene:clip:split"; split becomes a subdirectory so the arm jobs
# can point --cache and --val-cache at disjoint roots.
spec=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
IFS=':' read -r scene clip split <<< "$spec"
out="$OUT_ROOT/${split}/${scene}_c${clip}"
echo "task $SLURM_ARRAY_TASK_ID -> $scene clip $clip split $split -> $out"

mkdir -p "$out"
export TQDM_MININTERVAL=30
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh

# Use the interpreter directly, never `micromamba run`. `micromamba run` takes a
# lock under ~/.cache/mamba/proc, which lives on /home == CephFS, where locking is
# unreliable; a long-running job holds it for its whole lifetime and every later
# invocation blocks. Two timing jobs died at their walltime waiting on the lock
# held by two training arms.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"


"$PY_ENV" \
    .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
trap 'kill $! 2>/dev/null' EXIT

"$PY_ENV" scripts/memory/build_cache.py \
    --scannetpp-root "$ROOT" --scene "$scene" --clip-index "$clip" \
    --clip-len "$CLIP_LEN" --stride "$STRIDE" --out "$out"
