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

micromamba run -n lingbot_map python \
    .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
trap 'kill $! 2>/dev/null' EXIT

micromamba run -n lingbot_map python scripts/memory/build_cache.py \
    --scannetpp-root "$ROOT" --scene "$scene" --clip-index "$clip" \
    --clip-len "$CLIP_LEN" --stride "$STRIDE" --out "$out"
